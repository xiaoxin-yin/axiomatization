import re
import os
import sys
import json
import time
import signal
import shutil
import psutil
from pathlib import Path
from loguru import logger
from tempfile import mkdtemp
from shutil import ignore_patterns
from subprocess import TimeoutExpired
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Dict, Any, Optional

from ..constants import (
    TMP_DIR,
    TACTIC_CPU_LIMIT,
    TACTIC_MEMORY_LIMIT,
)
from ..utils import to_json_path
from .parse_goals import parse_goals, Goal
from ..data_extraction.trace import get_traced_repo_path
from ..data_extraction.lean import Theorem, LeanGitRepo, Pos
from ..container import get_container, Mount, NativeContainer, DockerContainer
from ..data_extraction.traced_data import TracedFile, get_code_without_comments


_REPL_PROMPT = "REPL>"


@dataclass(frozen=True)
class CommandState:
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)


@dataclass(frozen=True)
class TacticState:
    pp: str
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)
    goals: List[Goal] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        goals = parse_goals(self.pp)
        assert len(goals) == self.pp.count("⊢")
        object.__setattr__(self, "goals", goals)

    @property
    def num_goals(self) -> int:
        return len(self.goals)


@dataclass(frozen=True)
class ProofFinished:
    tactic_state_id: int
    message: Optional[str] = field(default=None, compare=False)


@dataclass(frozen=True)
class ProofGivenUp:
    pass


@dataclass(frozen=True)
class LeanError:
    error: str


@dataclass(frozen=True)
class TimeoutError:
    error: str


TacticResult = Union[
    TacticState,
    ProofFinished,
    LeanError,
    TimeoutError,
    ProofGivenUp,
]

CommandResult = Union[CommandState, LeanError, TimeoutError]

State = Union[CommandState, TacticState]


class DojoCrashError(Exception):
    @property
    def is_out_of_memory(self) -> bool:
        return str(self) == "OOM"


class DojoHardTimeoutError(Exception):
    pass


class DojoInitError(Exception):
    pass


def _kill_descendants(proc: psutil.Process) -> None:
    for child in proc.children():
        _kill_descendants(child)
    proc.kill()


class Dojo:
    """Gym-like environment for programmatic interaction with Lean through tactics or commands."""

    entry: Union[Theorem, Tuple[LeanGitRepo, Path, int]]
    hard_timeout: Optional[float]
    additional_imports: List[str]
    repo: LeanGitRepo
    file_path: Path
    is_successful: Optional[bool] = None
    is_crashed: bool = False
    has_timedout: bool = False

    def __init__(
        self,
        entry: Union[Theorem, Tuple[LeanGitRepo, Path, int]],
        hard_timeout: Optional[float] = None,
        additional_imports: List[str] = [],
        copy_tree_on_enter = True
    ):
        """Initialize Dojo.

        Args:
            entry (Union[Theorem, Tuple[LeanGitRepo, Path, int]]): When a Theorem is given,
                the :class:`Dojo` object enables interaction with the theorem through tactics.
                When a tuple of (repo, file_path, line_nb) is given (only supported in Lean 4),
                the :class:`Dojo` object enables interaction with Lean through commands (similar to a REPL).
            hard_timeout (Optional[float], optional): Hard timeout in seconds. Defaults to None.
        """
        self.entry = entry
        self.hard_timeout = hard_timeout
        self.additional_imports = additional_imports
        self.copy_tree_on_enter = copy_tree_on_enter
        self.tmp_dir = None
        self.proc = None

        if self.uses_tactics:
            assert isinstance(entry, Theorem)
            self.repo, self.file_path = entry.repo, entry.file_path
            self.is_successful = False
        else:
            assert self.uses_commands
            assert isinstance(entry, tuple)
            self.repo, self.file_path, _ = entry
            self.file_path = Path(self.file_path)

        if self.hard_timeout is None:
            logger.warning("Using Lean 4 without a hard timeout may hang indefinitely.")

    @property
    def uses_tactics(self) -> bool:
        return isinstance(self.entry, Theorem)

    @property
    def uses_commands(self) -> bool:
        return isinstance(self.entry, tuple)

    def __enter__(self, new_theorem_code=None, supporting_theorem_code="") -> Tuple["Dojo", State]:
        """Initialize Dojo."""
        logger.debug(f"Initializing Dojo for {self.entry}")

        should_copy_tree = False
        if self.copy_tree_on_enter or self.tmp_dir is None:
            # Work in a temporary directory.
            self.origin_dir = Path.cwd()
            self.tmp_dir = Path(mkdtemp(dir=TMP_DIR))
            should_copy_tree = True
            logger.debug(f"dojo.origin_dir={self.origin_dir}")
            logger.debug(f"Creating temp dir. dojo.tmp_dir={self.tmp_dir}")

        #try:
        self._install_handlers()
        os.chdir(self.tmp_dir)

        # Copy and `cd` into the repo.
        traced_repo_path = get_traced_repo_path(self.repo)
        if should_copy_tree:
            logger.info("Copy tree ...")
            shutil.copytree(
                traced_repo_path,
                self.repo.name,
                ignore=ignore_patterns("*.dep_paths", "*.ast.json", "*.trace.xml"),
            )

        os.chdir(self.repo.name)

        # Replace the human-written proof with a `repl` tactic.
        try:
            traced_file = self._locate_traced_file(traced_repo_path)
        except FileNotFoundError:
            raise DojoInitError(
                f"Cannot find the *.ast.json file for {self.entry} in {traced_repo_path}."
            )

        if new_theorem_code is None:
            self._modify_file(traced_file, append_lakefile=should_copy_tree)
        else:
            self._add_theorem(new_theorem_code, supporting_theorem_code, traced_file, append_lakefile=should_copy_tree)

        if should_copy_tree:
            logger.info("lake build Lean4Repl ...")
            # Run the modified file in a container.
            self.container = get_container()
            logger.info(f"Launching the proof using {type(self.container)}")
            mts = [Mount(Path.cwd(), Path(f"/workspace/{self.repo.name}"))]
            self.container.run(
                "lake build Lean4Repl",
                mts,
                as_current_user=True,
                capture_output=True,
                work_dir=f"/workspace/{self.repo.name}",
                cpu_limit=None,
                memory_limit=None,
                envs={},
            )
        else:
            mts = []

        assert re.fullmatch(r"\d+g", TACTIC_MEMORY_LIMIT)
        memory_limit = 1024 * int(TACTIC_MEMORY_LIMIT[:-1])
        cmd = f"lake env lean --threads={TACTIC_CPU_LIMIT} --memory={memory_limit} {self.file_path}"

        logger.info("run_interactive ...")
        if self.proc is not None:
            try:
                _kill_descendants(psutil.Process(self.proc.pid))
            except:
                pass
        self.proc = self.container.run_interactive(
            cmd,
            mts,
            cpu_limit=None,
            memory_limit=None,
            work_dir=f"/workspace/{self.repo.name}",
            as_current_user=True,
            envs={},
        )

        # Get the initial tactic state.
        try:
            stdout = self._read_next_line()
            #print("stdout:", stdout)
            res = json.loads(stdout[0])
        except Exception as ex:
            if traced_file.has_prelude:
                raise DojoInitError(
                    "Currently LeanDojo does not support interacting with proofs in prelude files."
                )
            elif isinstance(ex, EOFError):
                raise DojoInitError("EOF")
            else:
                raise ex

        assert res["error"] is None

        # logger.debug(f"Response: {res}")
        if self.uses_tactics:
            assert res["tacticState"] != "no goals"
            init_state: State = TacticState(
                self._post_process(res["tacticState"]),
                res["sid"],
            )
        else:
            assert self.uses_commands
            init_state = CommandState(int(res["sid"]))

        self.start_time = time.monotonic()
        self._set_timer()

        return self, init_state

#         except Exception as ex:
#             os.chdir(self.origin_dir)
#             shutil.rmtree(self.tmp_dir)
#             raise ex

    def _locate_traced_file(self, traced_repo_path: Path) -> TracedFile:
        json_path = to_json_path(traced_repo_path, self.file_path, self.repo)
        return TracedFile.from_traced_file(traced_repo_path, json_path, self.repo)

    def _set_timer(self) -> None:
        if self.hard_timeout is not None:
            signal.signal(signal.SIGALRM, self._handle_hard_timeout)
            signal.alarm(int(self.hard_timeout))

    def _cancel_timer(self) -> None:
        if self.hard_timeout is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def _handle_hard_timeout(self, signum: Any, frame: Any) -> None:
        logger.debug(f"Hard timeout in {self}")
        self.has_timedout = True
        raise DojoHardTimeoutError()

    def _install_handlers(self) -> None:
        self.old_sigint = signal.signal(signal.SIGINT, self._exit_gracefully)
        self.old_sigterm = signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _uninstall_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.old_sigint)
        signal.signal(signal.SIGTERM, self.old_sigterm)

    def _exit_gracefully(self, signum: Any, frame: Any) -> None:
        logger.debug("Exiting gracefully.")
        sys.exit(-1)

    def _cleanup(self) -> None:
        logger.debug("Cleaning up.")
        try:
            self._cleanup_container()
            self._cleanup_proc()
        finally:
            self._cleanup_tmp_dir()
            self._uninstall_handlers()

    def _cleanup_container(self) -> None:
        """Clean up the container."""
        logger.debug("Cleaning up the container.")
        assert isinstance(self.container, DockerContainer) or isinstance(
            self.container, NativeContainer
        )
        self.container.cleanup()

    def _cleanup_proc(self) -> None:
        """Clean up the subprocess."""
        logger.debug(f"Cleaning up the subprocess {self.proc.pid}.")
        _kill_descendants(psutil.Process(self.proc.pid))
        """
        self.proc.terminate()
        try:
            self.proc.wait(timeout=0.5)
        except TimeoutExpired:
            self.proc.kill()
        """

    def _cleanup_tmp_dir(self) -> None:
        """Clean up the temporary directory."""
        logger.debug("Cleaning up the temporary directory.")
        os.chdir(self.origin_dir)
        if self.tmp_dir is not None and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        """Exit Dojo.

        Args:
            exc_type (None): _description_
            exc_val (None): _description_
            exc_tb (None): _description_
        """
        # Cancel the hard timeout.
        self._cancel_timer()
        self._cleanup()

    def _post_process(self, tactic_state: str) -> str:
        """Post-process the pretty-printed tactic state.

        Args:
            tactic_state (str): _description_

        Returns:
            str: _description_
        """
        m = re.match(r"\d+ goals\n", tactic_state)
        if m is not None:
            return tactic_state[m.end() :]
        else:
            return tactic_state

    def _get_imports(self) -> str:
        imports = ["Lean4Repl"] + self.additional_imports
        return "\n".join(f"import {_}" for _ in imports) + "\n\n"

    def _add_theorem(self, code_thereom, code_supporting_theorem, traced_file: TracedFile, append_lakefile: True) -> None:
        logger.debug(f"Adding theorem to {traced_file.lean_file.path}")
        
#         assert isinstance(self.entry, Theorem)
#         traced_theorem = traced_file.get_traced_theorem(self.entry)
#         if traced_theorem is None:
#             raise DojoInitError(
#                 f"Failed to locate the theorem with `{self.entry.full_name}` as its fully qualified name"
#             )
#         proof_start, proof_end = traced_theorem.locate_proof()
        lean_file = traced_file.lean_file

#         code_import = self._get_imports()
        code_proof = "\nby\n  lean_dojo_repl\n  sorry\n"
        existing_code = get_code_without_comments(
            lean_file, lean_file.start_pos, lean_file.end_pos, traced_file.comments
        )
        modified_code = (
            self._get_imports()
            + existing_code
            + code_supporting_theorem
            + "\nset_option maxHeartbeats 0 in\n"
            + code_thereom
            + code_proof
        )
        #
        repl_file = "Lean4Repl.lean"
        repl_dst = Path(repl_file)
        
        if append_lakefile:
            if os.path.exists("lakefile.lean"):
                with open("lakefile.lean", "a") as oup:
                    oup.write("\nlean_lib Lean4Repl {\n\n}\n")
            else:
                assert os.path.exists("lakefile.toml")
                with open("lakefile.toml", "a") as oup:
                    oup.write('\n[[lean_lib]]\nname = "Lean4Repl"\n')

        if os.path.exists("lakefile.olean"):
            os.remove("lakefile.olean")
        if os.path.exists(".lake/lakefile.olean"):
            os.remove(".lake/lakefile.olean")

        # Copy the REPL code to the right directory.
        repl_src = Path(__file__).with_name(repl_file)
        repl_code = repl_src.open().read()
        #if repl_dst.exists():
        #    raise DojoInitError(f"{repl_dst} exists")
        with repl_dst.open("wt") as oup:
            oup.write(repl_code)

        # Write the modified code to the file.
        with self.file_path.open("wt") as oup:
            #print("MODIFIED_CODE\n", modified_code)
            oup.write(modified_code)
        
        if not append_lakefile:
            additional_file_path = f"{os.getcwd()}/workspace/{self.repo.name}/{self.file_path}"
            logger.info("Also copying to " + additional_file_path)
            additional_file_path = Path(additional_file_path)
            with additional_file_path.open("wt") as oup:
                oup.write(modified_code)
    
    def _modify_file(self, traced_file: TracedFile, append_lakefile: True) -> None:
        logger.debug(f"Modifying {traced_file.lean_file.path}")

        if self.uses_tactics:
            # Interaction through tactics.
            modified_code = self._modify_proof(traced_file)
        else:
            # Interaction through commands (supported only in Lean 4 via CommandElabM).
            lean_file = traced_file.lean_file
            pos = Pos(line_nb=self.entry[2], column_nb=1)
            code_before = get_code_without_comments(
                lean_file, lean_file.start_pos, pos, traced_file.comments
            )
            modified_code = (
                self._get_imports()
                + code_before
                + "set_option maxHeartbeats 0 in\n#lean_dojo_repl\n\n"
                + lean_file[pos:]
            )

        repl_file = "Lean4Repl.lean"
        repl_dst = Path(repl_file)

        if append_lakefile:
            if os.path.exists("lakefile.lean"):
                with open("lakefile.lean", "a") as oup:
                    oup.write("\nlean_lib Lean4Repl {\n\n}\n")
            else:
                assert os.path.exists("lakefile.toml")
                with open("lakefile.toml", "a") as oup:
                    oup.write('\n[[lean_lib]]\nname = "Lean4Repl"\n')

        if os.path.exists("lakefile.olean"):
            os.remove("lakefile.olean")
        if os.path.exists(".lake/lakefile.olean"):
            os.remove(".lake/lakefile.olean")

        # Copy the REPL code to the right directory.
        repl_src = Path(__file__).with_name(repl_file)
        repl_code = repl_src.open().read()
        if repl_dst.exists():
            logger.info("Removing existing " + str(repl_dst))
            os.remove(str(repl_dst))
            logger.info("Done removing " + str(repl_dst))
            #raise DojoInitError(f"{repl_dst} exists")
        with repl_dst.open("wt") as oup:
            oup.write(repl_code)

        # Write the modified code to the file.
        with self.file_path.open("wt") as oup:
            #print("MODIFIED_CODE\n", modified_code)
            oup.write(modified_code)
            
        #logger.info("_modify_file: All done. Writing to " + str(os.getcwd()) + "/" + str(self.file_path))
        if not append_lakefile:
            additional_file_path = f"{os.getcwd()}/workspace/{self.repo.name}/{self.file_path}"
            logger.info("Also copying to " + additional_file_path)
            additional_file_path = Path(additional_file_path)
            with additional_file_path.open("wt") as oup:
                oup.write(modified_code)

    def _modify_proof(self, traced_file: TracedFile) -> str:
        # Modify the proof and set up the `repl` tactic.
        assert isinstance(self.entry, Theorem)
        traced_theorem = traced_file.get_traced_theorem(self.entry)
        if traced_theorem is None:
            raise DojoInitError(
                f"Failed to locate the theorem with `{self.entry.full_name}` as its fully qualified name"
            )
        proof_start, proof_end = traced_theorem.locate_proof()
        lean_file = traced_file.lean_file

        code_import = self._get_imports()
        code_proof = "\nby\n  lean_dojo_repl\n  sorry\n"
        code_before_theorem = get_code_without_comments(
            lean_file, lean_file.start_pos, traced_theorem.start, traced_file.comments
        )
        code_thereom = lean_file[traced_theorem.start : proof_start]
        modified_code = (
            code_import
            + code_before_theorem
            + "\nset_option maxHeartbeats 0 in\n"
            + code_thereom
            + code_proof
            + lean_file[proof_end:]
        )

        return str(modified_code)

    def run_tac(self, state: TacticState, tactic: str) -> TacticResult:
        if not isinstance(state, TacticState):
            raise RuntimeError(
                f"Attempting to run a tactic on an invalid state {state}."
            )
        assert isinstance(tactic, str), f"Invalid tactic {tactic}"

        tsid = state.id
        req = json.dumps({"sid": tsid, "cmd": tactic}, ensure_ascii=False)
        res = self._submit_request(req)

        if res["error"] is not None:
            if "proof contains `sorry`" in res["error"]:
                return ProofGivenUp()
            elif "try_for_time tactic failed, timeout" in res["error"]:
                return TimeoutError(res["error"].strip())
            else:
                return LeanError(res["error"].strip())
        elif res["tacticState"] == "no goals":
            self.is_successful = True
            return ProofFinished(res["sid"], res["message"])
        else:
            tactic_state = self._post_process(res["tacticState"])
            return TacticState(
                tactic_state,
                res["sid"],
                res["message"],
            )

    def run_cmd(self, state: CommandState, command: str) -> CommandResult:
        if not isinstance(state, CommandState):
            raise RuntimeError(
                f"Attempting to run a command on an invalid state {state}."
            )
        assert isinstance(command, str), f"Invalid command {command}"

        csid = state.id
        req = json.dumps({"sid": csid, "cmd": command}, ensure_ascii=False)
        res = self._submit_request(req)

        if res["error"] is not None:
            return LeanError(res["error"].strip())
        else:
            return CommandState(res["sid"], res["message"])

    def _submit_request(self, req: str) -> Dict[str, Any]:
        """Submit a request to Lean and get the response.

        Args:
            req (str): _description_

        Raises:
            DojoCrashError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        if self.proc.stdin is None:
            raise RuntimeError("self.proc.stdin is not initialized")
        self._check_alive()
        logger.debug(req)
        self.proc.stdin.write(req + "\n")
        try:
            res, msg = self._read_next_line()
        except EOFError:
            raise DojoCrashError("Unexpected EOF")
        try:
            result: Dict[str, Any] = json.loads(res)
        except json.decoder.JSONDecodeError:
            raise DojoCrashError(f"Invalid JSON: {res}")

        result["message"] = msg
        return result

    def _check_alive(self) -> None:
        exit_code = self.proc.poll()
        if exit_code is None:
            return
        elif exit_code == 137:
            raise DojoCrashError("OOM")
        else:
            raise DojoCrashError(f"Unknown exit code: {exit_code}")

    def _read_next_line(self) -> Tuple[str, str]:
        """Read the next line from `self.proc`.

        Raises:
            EOFError: _description_
            DojoCrashError: _description_
            DojoInitError: _description_

        Returns:
            str: _description_
        """
        if self.proc.stdout is None:
            raise RuntimeError("self.proc.stout is not initialized")
        msg: List[str] = []
        while True:
            line = self.proc.stdout.readline().strip()
            print("NEXTLINE:", line)
            logger.debug(line)
            if line == "":
                raise EOFError
            if line.startswith(_REPL_PROMPT):
                self._check_alive()
                return line[len(_REPL_PROMPT) :].strip(), "\n".join(msg)
            elif "error: " in line:
                if (
                    "error: deep recursion was detected" in line
                    or "error: [fatal] not_a_theorem" in line
                ):
                    self.is_crashed = True
                    raise DojoCrashError(line)
                elif "error: unknown package" in line:
                    self.is_crashed = True
                    raise DojoInitError(line)
                else:
                    pass
            else:
                msg.append(line)

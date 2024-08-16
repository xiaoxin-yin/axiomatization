import re
from collections import OrderedDict, defaultdict

class Premise:
    
    def __init__(self, theorem_text=None):
        self.full_paht = None
        self.theorem_text = theorem_text
        self.theorem_name = ""
        self.hypotheses = OrderedDict()
        self.target = ""
        self.tactics = []
        if theorem_text is not None:
            self.parse()
        self.state = None
        
    def clone(self):
        new_premise = Premise()
        new_premise.theorem_text = self.theorem_text
        new_premise.theorem_name = self.theorem_name
        new_premise.hypotheses = OrderedDict(self.hypotheses)
        new_premise.target = self.target
        new_premise.tactics = list(self.tactics)
        return new_premise

    def parse(self):
        # Split the theorem text into parts
        parts = re.split(r'(:=)', self.theorem_text, 1)
        
        # Parse the theorem name, hypotheses, and target
        self.parse_header(parts[0])
        
        self.tactics = self.parse_tactics(self.theorem_text)

    def parse_state(self, state_pp):
        hypotheses = OrderedDict()
        target = ""

        # Split the state into lines
        lines = state_pp.strip().split('\n')

        for line in lines:
            # Check if the line defines variables
            if ':' in line:
                idx = line.index(':')
                var_def = (line[0:idx], line[idx+1:])
                vars = var_def[0].strip().split()
                type = var_def[1].strip()
                for var in vars:
                    hypotheses[var] = type
            # Check if the line is the target
            elif line.strip().startswith('⊢'):
                target = line.strip().split('⊢')[1].strip()

        self.hypotheses = hypotheses
        self.target = target
        
        
    def parse_header(self, header):
        # Extract theorem name
        name_match = re.match(r'(theorem|lemma)\s+(\w+)', header)
        if name_match:
            self.theorem_name = name_match.group(2)
            header = header[name_match.end():].strip()
        
        # Extract hypotheses
        while header.startswith('(') or header.startswith('[') or header.startswith('{'):
            end_index = self.find_matching_bracket(header)
            if end_index == -1:
                break
            self.parse_hypothesis(header[1:end_index])
            header = header[end_index + 1:].strip()
        
        # Extract target
        if ':' in header:
            self.target = header.split(':', 1)[1].strip()

    def find_matching_bracket(self, text):
        stack = []
        for i, char in enumerate(text):
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack:
                    return -1
                if (char == ')' and stack[-1] == '(') or \
                   (char == ']' and stack[-1] == '[') or \
                   (char == '}' and stack[-1] == '{'):
                    stack.pop()
                    if not stack:
                        return i
        return -1

    def parse_hypothesis(self, hypothesis):
        parts = hypothesis.split(':')
        if len(parts) == 1:
            key, value = parts[0].split(None, 1)
            self.hypotheses[key.strip()] = None if value.strip() == 'Type*' else value.strip()
        else:
            vars, type_or_condition = parts
            vars = vars.split()
            for var in vars:
                self.hypotheses[var.strip()] = type_or_condition.strip()

    def parse_tactics(self, theorem_text):
        # Split the text at ":="
        parts = theorem_text.split(":=", 1)
        
        # If ":=" is not found, return an empty list
        if len(parts) < 2:
            return []
        
        # Get the part after ":="
        tactics_part = parts[1].strip()
        
        # Remove "by" if it's present at the start
        if tactics_part.lower().startswith("by"):
            tactics_part = tactics_part[2:].strip()
        
        # Split the tactics by newline and strip whitespace
        tactics = [tactic.strip() for tactic in tactics_part.split("\n")]
        
        # Remove any empty tactics
        tactics = [tactic for tactic in tactics if tactic]
        
        return tactics

    def to_theorem_code(self, include_tactics=False):
        # Start with the theorem type and name
        result = f"theorem {self.theorem_name} "
        
        # Group hypotheses with the same value
        grouped_hypotheses = {}
        for key, value in self.hypotheses.items():
            if value not in grouped_hypotheses:
                grouped_hypotheses[value] = []
            grouped_hypotheses[value].append(key)
        
        # Add hypotheses
        for value, keys in grouped_hypotheses.items():
            if value is None:
                result += f"[{' '.join(keys)}] "
            else:
                result += f"({' '.join(keys)}: {value}) "
        
        # Add target
        result += f": {self.target} := "
        
        if include_tactics:
            # Add tactics
            if self.tactics:
                result += "by\n  " + "\n  ".join(self.tactics)
        
        return result.strip()

    def __str__(self):
        return f"Theorem Name: {self.theorem_name}\n" \
               f"Hypotheses: {self.hypotheses}\n" \
               f"Target: {self.target}\n" \
               f"Tactics: {self.tactics}"


def extract_variable_names(expression):
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    variables = set(re.findall(pattern, expression))
    return list(variables)

    
# Returns a tuple of three: (1) A list of used variable/hypothesis names, (2) A new variable name, (3) A new hypothesis name
def get_next_variable_name(premise, other_names=[]):
    # Get all hypothesis names
    used_names = set(list(premise.hypotheses.keys()) + other_names)
    
    # Generate all lowercase letters
    all_letters = set(chr(i) for i in range(ord('a'), ord('z') + 1) if i != ord('h'))
    
    # Find unused variable name
    unused_letters = all_letters - used_names
    next_letter = None
    if unused_letters:
        # If there are unused letters, return the first one alphabetically
        next_letter = min(unused_letters)
    else:
        # If all letters are used, generate names like "a1", "a2", etc.
        i = 1
        while True:
            for letter in all_letters:
                new_name = f"{letter}{i}"
                if new_name not in used_names:
                    next_letter = new_name
                    break
            i += 1
    
    # Find unused hypothesis name
    for i in range(1, 10000):
        if f"h{i}" not in used_names:
            next_hyp_name = f"h{i}"
            break
    
    return list(premise.hypotheses.keys()), next_letter, next_hyp_name


def tokenize(text):
    pattern = r'([(),]|\s+)'
    tokens = re.split(pattern, text)
    return [token for token in tokens if token.strip()]


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

class TargetNode:
    operators = [','] + \
        ['∀', '∃'] + \
        ['↦', '→', '↔', '=>'] + \
        ['=', '≃', '≠', '<', '>', '≤', '≥'] + \
        ['∪', '∩', '⊆', '⊂', '⊇', '⊃', '\\'] + \
        ['+', '-', '*', '×', '/', '∈', '∉']
    left_parentheses =  ['(', '[', '{']
    right_parentheses = [')', ']', '}']

    def __init__(self, value, left=None, right=None, tokens=None, spliter_index=None):
        self.value = value
        self.left = left
        self.right = right
        self.tokens = tokens
        self.spliter_index = spliter_index
        
    def print_tree(self, level=0):
        print("  " * level + str(self.value))
        if self.left:
            self.left.print_tree(level + 1)
        if self.right:
            self.right.print_tree(level + 1)
            
    def to_string(self):
        if self.tokens is None:
            return ''
        return ' '.join(list(self.tokens))
    
    def is_operator_qualified(self, operators):
        SYNONYMOUS_OPERATORS = defaultdict(list)
        SYNONYMOUS_OPERATORS['*'] = ['×']
        operator_qualified = False
        if operators is None or len(operators) == 0:
            operator_qualified = True
        else:
            if self.value is not None and len(self.value) > 0:
                operator = self.value[0]
                if operator == operators[0] or \
                    operator in SYNONYMOUS_OPERATORS[operators[0]]:
                    operator_qualified = True
        return operator_qualified
    
    # Get operands for a theorem. 
    # For example, mul_add is defined as "a * (b + c) = a * b + a * c",
    # which has 3 operands and the operators is ["*",'+'].
    # When this function sees a "*" or '×', it will try to either get two operands
    # from the left-hand side and one from the right-hand side, or vice versa.
    # Returns a list of tuples, in which each tuple is a group of possible operands
    def __get_operands__(self, num_operands, operators):
        #print("GET_OPERANDS:", num_operands, operators, self.to_string())
        if num_operands <= 0:
            #print([])
            return []
        elif num_operands == 1:
            # Single operand
            operator_qualified = self.is_operator_qualified(operators)
            if operator_qualified:
                if operators is None or len(operators) == 0:
                    operands = set([self.to_string()])
                    if len(self.tokens) > 1:
                        for token in self.tokens:
                            if TargetNode.is_variable(token):
                                operands.add(token)
                    #print([tuple([x]) for x in operands])
                    return [tuple([x]) for x in operands]
                else:
                    sub_operators = operators[1:]
                    operands = set()
                    if self.left is not None:
                        operands.add(tuple([self.left.to_string()]))
                        operands.update(self.left.__get_operands__(1, sub_operators))
                    if self.right is not None:
                        operands.add(tuple([self.right.to_string()]))
                        operands.update(self.right.__get_operands__(1, sub_operators))
                    #print(list(operands))
                    return list(operands)
            else:
                operands = set()
                if self.left is not None:
                    operands.update(self.left.__get_operands__(1, operators))
                if self.right is not None:
                    operands.update(self.right.__get_operands__(1, operators))
                #print(list(operands))
                return list(operands)
        else:
            # Multiple operand
            operator_qualified = self.is_operator_qualified(operators)
            if not operator_qualified:
                output = set()
                if self.left is not None:
                    output.update(self.left.__get_operands__(num_operands, operators))
                if self.right is not None:
                    output.update(self.right.__get_operands__(num_operands, operators))
                #print(list(output))
                return list(output)
            else:
                if self.left is None or self.right is None:
                    return []
                output = set()
                sub_operators = None if operators is None else operators[1:]
                for num_left in range(1, num_operands):
                    num_right = num_operands - num_left
                    # Send sub_operators to the left
                    left_operands  = self.left.__get_operands__(num_left, sub_operators)
                    right_operands = self.right.__get_operands__(num_right, None)
                    for op1 in left_operands:
                        for op2 in right_operands:
                            output.add(tuple(list(op1) + list(op2)))
                            output.add(tuple(list(op2) + list(op1)))
                    # Send sub_operators to the right
                    left_operands  = self.left.__get_operands__(num_left, None)
                    right_operands = self.right.__get_operands__(num_right, sub_operators)
                    for op1 in left_operands:
                        for op2 in right_operands:
                            output.add(tuple(list(op1) + list(op2)))
                            output.add(tuple(list(op2) + list(op1)))
                #print(list(output))
                return list(output)
        
    def get_operands(self, num_operands, operators):
        operands = self.__get_operands__(num_operands, operators)
        for i in range(len(operands)):
            operand = list(operands[i])
            for j in range(len(operand)):
                if ' ' in operand[j]:
                    operand[j] = f"({operand[j]})"
            operands[i] = tuple(operand)
        return operands
    
    @classmethod
    def is_operator(cls, token):
        if token[0] in cls.operators:
            return True

    @classmethod
    def is_variable(cls, token):
        if cls.is_operator(token):
            return False
        if is_float(token):
            return False
        if token in cls.left_parentheses or token in cls.right_parentheses:
            return False
        return True
        
    @classmethod
    def get_operator_priority(cls, token):
        if token in cls.operators:
            return cls.operators.index(token)
        else:
            return cls.operators.index(token[0])

    @classmethod
    def find_lowest_priority_operator(cls, tokens):
        lowest_priority = float('inf')
        lowest_index = -1
        paren_count = 0

        for i, token in enumerate(tokens):
            if token in cls.left_parentheses:
                paren_count += 1
            elif token in cls.right_parentheses:
                paren_count -= 1
            elif paren_count == 0 and cls.is_operator(token):
                priority = cls.get_operator_priority(token)
                if priority <= lowest_priority:
                    lowest_priority = priority
                    lowest_index = i

        return lowest_index

    @classmethod
    def build_tree(cls, tokens):
        if not tokens:
            return None    

        index = cls.find_lowest_priority_operator(tokens)
        if index == -1:
            if tokens[0] in cls.left_parentheses and tokens[-1] in cls.right_parentheses:
                return cls.build_tree(tokens[1:-1])
            return TargetNode(' '.join(tokens), None, None, tokens, index)

        operator = tokens[index]
        left_tokens = tokens[:index]
        right_tokens = tokens[index+1:]

        return TargetNode(operator, cls.build_tree(left_tokens), cls.build_tree(right_tokens), tokens, index)
        
         
# If target is an equation or inequation, return left part, operator, and right parts. Otherwise return None
def split_equation(target):
    tokens = tokenize(target)
    root = TargetNode.build_tree(tokens)
    if root is None or root.value == '' or root.left is None or root.right is None:
        return None
    if root.value[0] in ['=', '≃', '≠', '<', '>', '≤', '≥']:
        return (root.left.to_string(), str(root.value), root.right.to_string())
        



def main() -> int:
    # Test cases for Premise class
    test_cases = [
        "theorem test_theorem_1 (a b: ℝ ) (h : c = b * a - d): a * b + a * b = 2 * a * b := by ring ",
        "lemma not_monotone_iff {f : ℝ → ℝ} : ¬Monotone f ↔ ∃ x y, x ≤ y ∧ f x > f y := by rw [Monotone]\n push_neg\n rfl",
        """theorem subgroup_closure {G : Type*} [Group G] (H H' : Subgroup G) : ((H ⊔ H' : Subgroup G) : Set G) = Subgroup.closure ((H : Set G) ∪ (H' : Set G)) := by
      rw [Subgroup.sup_eq_closure]
      rfl"""
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"Case {i}:")
        premise = Premise(case)
        print(premise)
        print(premise.to_theorem_code(include_tactics=True))
        print()

    print("Case parse_state")
    premise = Premise()
    premise.parse_state("""a b c d : ℝ
    h : a = c + d
    ⊢ b + c - (b + c) = 0""")
    print(premise.to_theorem_code())


    # Test cases for TargetNode
    targets = [
        "∀ x y, x = y → f x = f y",
        "∃ x, (f x = g x) => Prime x",
        "u ×₃ (v ×₃ w) + v ×₃ (w ×₃ u) + w ×₃ (u ×₃ v) = 0",
        "s \ t ∪ t \ s = (s ∪ t) \ (s ∩ t)",
        "(⋃ p ∈ primes, { x | p ^ 2 ∣ x }) = { x | ∃ p ∈ primes, p ^ 2 ∣ x } ",
        "c * (a * b + a * b) = c * (2 * a * b)"
    ]

    for target in targets:
        print(target)
        tokens = tokenize(target)
        print(tokens)
        root = TargetNode.build_tree(tokens)
        root.print_tree()
        print()

    expected_split_equation = [
        None,
        None,
        ('u ×₃ ( v ×₃ w ) + v ×₃ ( w ×₃ u ) + w ×₃ ( u ×₃ v )', '=', '0'),
        ('s \\ t ∪ t \\ s', '=', '( s ∪ t ) \\ ( s ∩ t )'),
        ('⋃ p ∈ primes , { x | p ^ 2 ∣ x }', '=', 'x | ∃ p ∈ primes , p ^ 2 ∣ x'),
        ('c * ( a * b + a * b )', '=', 'c * ( 2 * a * b )')
    ]

    print("Testing split_equation")
    for i in range(len(targets)):
        assert split_equation(targets[i]) == expected_split_equation[i]
    print("Done")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
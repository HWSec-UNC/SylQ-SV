"""Helpers for working with Z3, specifically parsing the symbolic expressions into 
Z3 expressions and solving for assertion violations."""

import z3
from z3 import Solver, Int, BitVec, Context, BitVecSort, ExprRef, BitVecRef, If, BitVecVal, And, IntVal, Int2BV, Or, Not, ULT, UGT, Z3Exception, BoolRef
from z3 import is_and, is_app_of, Z3_OP_EXTRACT, is_eq, is_distinct
from helpers.rvalue_parser import parse_tokens, tokenize
from engine.execution_manager import ExecutionManager
from engine.symbolic_state import SymbolicState
import pyslang as ps
import networkx as nx
import ast
from copy import deepcopy


BINARY_OPS = ("Plus", "Minus", "Power", "Times", "Divide", "Mod", "Sll", "Srl", "Sla", "Sra", "LessThan",
"GreaterThan", "LessEq", "GreaterEq", "Eq", "NotEq", "Eql", "NotEql", "And", "Xor",
"Xnor", "Or", "Land", "Lor")
op_map = {"Plus": "+", "Minus": "-", "Power": "**", "Times": "*", "Divide": "/", "Mod": "%", "Sll": "<<", "Srl": ">>>",
"Sra": ">>", "LessThan": "<", "GreaterThan": ">", "LessEq": "<=", "GreaterEq": ">=", "Eq": "=", "NotEq": "!=", "Eql": "===", "NotEql": "!==",
"And": "&", "Xor": "^", "Xnor": "<->", "Land": "&&", "Lor": "||"}

class Z3Visitor():
    def __init__(self, prefix):
        """Constructor that sets the prefix for variable names."""
        self.prefix = prefix
        print("prefix", prefix)
        #self.visited_nodes = set() 

    def visit(self, node):
        """A visitor that processes the node to generate Z3 expressions."""
        print(f"Visiting node: {node}") 
        print(f"Visiting node Type: {type(node)}")  
        if isinstance(node, ps.Token):
            result = self.handle_token(node)
        elif isinstance(node, ps.IdentifierNameSyntax):
            result = self.handle_identifier(node)
        elif isinstance(node, ps.IdentifierSelectNameSyntax):
            result = self.handle_identifier_select_name(node)
        elif isinstance(node, ps.ElementSelectSyntax):
            result = self.handle_element_select(node)
        elif isinstance(node, ps.BinaryExpressionSyntax):
            result = self.handle_binary_expression(node)
        elif isinstance(node, ps.ParenthesizedExpressionSyntax):
            result = self.handle_parenthesized_expression(node)
            print("result", type(result))
        elif isinstance(node, ps.LiteralExpressionSyntax):
            result = self.handle_literal_expression(node)
        elif isinstance(node, ps.BitSelectSyntax):
            result = self.handle_bit_select(node)
        elif isinstance(node, ps.ScopedNameSyntax):
            result = self.handle_scoped_name(node)
        elif isinstance(node, ps.IntegerVectorExpressionSyntax):
            result = self.handle_integer_vector_expression(node)
        elif isinstance(node, ps.PrefixUnaryExpressionSyntax):
            result = self.handle_prefix_unary_expression(node)
        else:
            print(f"Unhandled syntax: {type(node)}")
            return None
        print(result)
        if isinstance(result, ps.VisitAction):
            print(f"Encountered VisitAction: {result}")
            return None  
        return result

    def handle_integer_vector_expression(self, node):
        """Handle integer vector expressions."""
        print(f"Handling IntegerVectorExpression: {node}")
        
        print("Attributes of the node:", dir(node))

        if hasattr(node, 'value'):
            value = node.value  
            print(f"Value of the IntegerVectorExpression: {value}")
            return BitVecVal(int(str(value)), 32)  #

        elif hasattr(node, 'size'):
            size = node.size 
            print(f"Size of the IntegerVectorExpression: {size}")
            return BitVecVal(int(str(size)), 32)  
        return None   

    def handle_identifier(self, node):
        """Handle identifiers."""
        print(f"Handling identifier: {str(node.identifier)}")
        variable = str(node.identifier)
        return BitVec(variable, 32)
    
    def handle_identifier_select_name(self, node):
        """Handle indexed or array accesses like 'match[i]'."""
        print(f"Handling identifier select: {str(node.identifier)}[{node.selectors}]")
        
        # Extract the identifier ('match' or 'conf_i')
        identifier = str(node.identifier)
        
        # Get the index, assuming it's the first selector for example  'match[i]', i will be the selector)
        index_expr = self.visit(node.selectors[0])  
        print("index_expr",type(index_expr))
        index_val = int(str(index_expr))  
        variable = f"{identifier}[{index_val}]" 
        print("Fully Verified Variable:", variable)
        return BitVec(variable, 32)
 
    def handle_scoped_name(self, node):
            """Handle scoped names, including indexed names like conf_i[i].locked."""
            print(f"Handling scoped name: {node}")
            
            if str(node.separator) == "::":
                # Scoped names like riscv::PRIV_LVL_M
                scoped_name = str(node)
                return BitVec(scoped_name, 32)
            
            elif str(node.separator) == ".":
                # Field access like conf_i[i].locked
                # First, handle the base (conf_i[i])
                base = self.visit(node.left)  # Conf_i[i]
                print("base",base)
                # Then handle the field (locked)
                field = str(node.right)  # Field access (locked)
                variable= str(f"{base}[{field}]")
                return BitVec(variable, 32)

    def handle_element_select(self, node):
        """Handle element selection like structs and arrays."""
        print(f"Handling element select: {node}")
        element = self.visit(node.selector)  
        return element
    

    def handle_bit_select(self, node):
        """Handle bit select expressions like 'match[i]'."""
        print(f"Handling bit select expression: {node}")

       
        return BitVec(f"{node}", 32)

    def handle_literal_expression(self, node):
        """Handle literal expressions."""
        print(f"Handling literal expression: {node}")
        literal_value = node  
        if literal_value == 0:
            return BitVecVal(0, 32)  
        return BitVecVal(int(str(literal_value)), 32)  

    def convert_bitvec_to_bool(self, bitvec_expr):
        """Converts a BitVec expression to a Boolean (True if non-zero, False if zero)."""
        return UGT(bitvec_expr, BitVecVal(0, 32))

    def handle_prefix_unary_expression(self, node):
        """Handle prefix unary expressions (like NOT)."""
        print(f"Handling prefix unary expression: {node}")
        operator = str(node.operatorToken).strip()
        operand = self.visit(node.operand)
        if operator == "!":
            return Not(operand)
        elif operator == "-":
            return -operand
        else:
            print(f"Unsupported unary operator: {operator}")
            raise ValueError(f"Unsupported unary operator: {operator}")


    def handle_binary_expression(self, node):
        """Handle binary expressions (AND, OR, equality, etc.)."""
        print(f"Handling binary expression: {node.operatorToken}")
        left_expr = self.visit(node.left)
        print("done")
        right_expr = self.visit(node.right)
        print("done2")
        operator = str(node.operatorToken).strip()

        # issue
        print((left_expr))
        print(node.left)
        if str(left_expr.sort()) == "Bool" and str(right_expr.sort()) != "Bool":
            right_expr = UGT(right_expr, BitVecVal(0, 32)) 
            print(f"Converted Right Expression to Bool: {right_expr}")

        print(operator)
        print(node.left)
        print(node.right)
        print(left_expr.sort())
        print(right_expr.sort())
        if operator == "==":
            return left_expr == right_expr
        elif operator == "!=":
            return left_expr != right_expr
        elif operator == "&&":
            return And(left_expr, right_expr)
        elif operator == "||":
            return Or(left_expr, right_expr)
        elif operator == ">":
            return UGT(left_expr, right_expr) 
        elif operator == "<":
            return ULT(left_expr, right_expr) 
        elif isinstance(left_expr, BitVecRef) and isinstance(right_expr, BitVecRef):
            return UGT(left_expr, BitVecVal(0, 32)) == right_expr
        
        else:
            print(f"Unsupported binary operator: {operator}")
            raise ValueError(f"Unsupported binary operator: {operator}")


    def handle_parenthesized_expression(self, node):
        """Handle parenthesized expressions."""
        print("Handling parenthesized expression.")
        return (self.visit(node.expression))
    
    def get_full_variable_name(self,variable):
        """Generate the full variable name by appending the variable to the prefix."""
        return f"{self.prefix}.{variable}"
    
def pyslang_to_z3(expr, prefix=""):
    """Parse the expression and convert it into a Z3 expression."""
    print(f"Parsing expression: {expr}")
    syntax_tree = ps.SyntaxTree.fromText(expr)
    root = syntax_tree.root
    visitor = Z3Visitor(prefix)
    z3_expression = visitor.visit(root)    
    return z3_expression


def get_constants_list(new_constraint, s: SymbolicState, m: ExecutionManager):
    """Get list of constants that need to be added to z3 context from symbolic expressions."""
    res = []
    words = new_constraint.split(" ")
    for word in words:
        if word in s.store[m.curr_module].values():
            res.append(word)
    return res

def parse_concat_to_Z3(concat, s: SymbolicState, m: ExecutionManager):
    """Takes a concatenation of symbolic symbols areturns the list of bitvectors"""
    res = []
    for key in concat:
        x = BitVec(concat[key], 1)
        res.append(x)
    return res


def parse_expr_to_Z3(e: ps.ExpressionSyntax, s: SymbolicState, m: ExecutionManager):
    """Takes in a complex Verilog Expression and converts it to 
    a Z3 query."""
    tokens_list = parse_tokens(tokenize(e, s, m))
    new_constraint = evaluate_expr(tokens_list, s, m)
    new_constants = []
    if not new_constraint is None: 
        new_constants = get_constants_list(new_constraint, s, m)
    if is_and(e):
        lhs = parse_expr_to_Z3(e.left, s, m)
        rhs = parse_expr_to_Z3(e.right, s, m)
        return s.pc.add(lhs.assertions() and rhs.assertions())
    elif is_app_of(e, Z3_OP_EXTRACT):
        part_sel_expr = f"{e.var.name}[{e.msb}:{e.lsb}]"
        module_name = m.curr_module
        is_reg = e.var.name in m.reg_decls
        if not e.var.scope is None:
            module_name = e.scope.labellist[0].name
        if s.store[module_name][e.var.name].isdigit():
            int_val = IntVal(int(s.store[module_name][e.name]))
            return Int2BV(int_val, 32)
        else:
            if not part_sel_expr in s.store[m.curr_module] and "[" in part_sel_expr:
                parts = part_sel_expr.partition("[")
                first_part = parts[0]
                s.store[m.curr_module][part_sel_expr] = s.store[m.curr_module][first_part]
            return BitVec(s.store[module_name][part_sel_expr], 32)
    elif e.__class__.__name__ == "IdentifierNameSyntax":
        module_name = m.curr_module  # Default to current module
        # PySlang 7.0 IdentifierNameSyntax uses .identifier.valueText for the name
        # Access the identifier name through .identifier attribute
        if not hasattr(e, "identifier"):
            # Fallback: try to get name directly if identifier attribute doesn't exist
            var_name = getattr(e, "valueText", getattr(e, "name", None))
            if var_name is None:
                return BitVecVal(0, 32)  # Return zero if we can't get the name
        else:
            var_name = e.identifier.valueText if hasattr(e.identifier, "valueText") else None
            if var_name is None:
                var_name = getattr(e.identifier, "name", None)
        
        if var_name is None:
            return BitVecVal(0, 32)  # Return zero if we can't get the name
            
        is_reg = var_name in m.reg_decls if hasattr(m, "reg_decls") else False
        
        # Check if variable exists in store, if not return zero
        if module_name not in s.store or var_name not in s.store[module_name]:
            return BitVecVal(0, 32)
            
        if s.store[module_name][var_name].isdigit():
            int_val = IntVal(int(s.store[module_name][var_name]))
            return Int2BV(int_val, 32)
        else:
            return BitVec(s.store[module_name][var_name], 32)
    elif e.__class__.__name__ == "IntegerLiteralExpressionSyntax":
        int_val = IntVal(e.value)
        return Int2BV(int_val, 32)
    elif is_eq(e):
        lhs = parse_expr_to_Z3(e.left, s, m)
        rhs = parse_expr_to_Z3(e.right, s, m)
        if m.branch:
            s.pc.add(lhs == rhs)
        else:
            s.pc.add(lhs != rhs)
        return (lhs == rhs)
    elif is_distinct(e):
        lhs = parse_expr_to_Z3(e.left, s, m)
        rhs = parse_expr_to_Z3(e.right, s, m)
        if m.branch:          
            # only RHS is BitVec (Lhs is a more complex expr)
            if isinstance(rhs, z3.z3.BitVecRef) and not isinstance(lhs, z3.z3.BitVecRef):
                c = If(lhs, BitVecVal(1, 32), BitVecVal(0, 32))
                s.pc.add(c != rhs)
            else:
                s.pc.add(lhs != rhs)
        else:
            # only RHS is bitVEC 
            if isinstance(rhs, z3.z3.BitVecRef) and not isinstance(lhs, z3.z3.BitVecRef):
                c = If(lhs, BitVecVal(1, 32), BitVecVal(0, 32))
                #print("a")
                s.pc.add(c == rhs)
            else:
                s.pc.push()
                s.pc.add(lhs == rhs)
                if not solve_pc(s.pc):
                    s.pc.pop()
                    m.abandon = True
                    m.ignore = True
    elif is_and(e):
        lhs = parse_expr_to_Z3(e.left, s, m)
        rhs = parse_expr_to_Z3(e.right, s, m)

        if isinstance(rhs, BitVecRef) and isinstance(lhs, BitVecRef):
            return s
        elif isinstance(rhs, BitVecRef):
            return s
        elif isinstance(lhs, BitVecRef):
            return s
        else:
            if lhs is None:
                return s.pc.add(rhs.pc.assertions())
            
            if rhs is None:
                return s.pc.add(rhs.pc.assertions())

            return s
    return s

def solve_pc(s: Solver) -> bool:
    """Solve path condition."""
    result = str(s.check())
    if str(result) == "sat":
        model = s.model()
        return True
    else:
        print("unsat")
        print(s)
        print(s.unsat_core())
        return False

def evaluate_expr(parsedList, s: SymbolicState, m: ExecutionManager):
    for i in parsedList:
        res = eval_expr(i, s, m)
    return res

def evaluate_expr_to_smt(lhs, rhs, op, s: SymbolicState, m: ExecutionManager) -> str: 
    """Helper function to resolve binary symbolic expressions."""
    if (isinstance(lhs,tuple) and isinstance(rhs,tuple)):
        return f"({op} ({eval_expr(lhs, s, m)})  ({eval_expr(rhs, s, m)}))"
    elif (isinstance(lhs,tuple)):
        if (isinstance(rhs,str)) and not rhs.isdigit():
            return f"({op} ({eval_expr(lhs, s, m)}) {s.get_symbolic_expr(m.curr_module, rhs)})"
        else:
            return f"({op} ({eval_expr(lhs, s, m)}) {str(rhs)})"
    elif (isinstance(rhs,tuple)):
        if (isinstance(lhs,str)) and not lhs.isdigit():
            return f"({op} ({s.get_symbolic_expr(m.curr_module, lhs)}) ({eval_expr(rhs, s, m)}))"
        else:
            return f"({op} {str(lhs)}  ({eval_expr(rhs, s, m)}))"
    else:
        if (isinstance(lhs ,str) and isinstance(rhs , str)) and not lhs.isdigit() and not rhs.isdigit():
            return f"({op} {s.get_symbolic_expr(m.curr_module, lhs)} {s.get_symbolic_expr(m.curr_module, rhs)})"
        elif (isinstance(lhs ,str)) and not lhs.isdigit():
            return f"({op} {s.get_symbolic_expr(m.curr_module, lhs)} {str(rhs)})"
        elif (isinstance(rhs ,str)) and not rhs.isdigit():
            return f"({op} {str(lhs)}  {s.get_symbolic_expr(m.curr_module, rhs)})"
        else: 
            return f"({op} {str(lhs)} {str(rhs)})"
 
def eval_expr(expr, s: SymbolicState, m: ExecutionManager) -> str:
    """Takes in an AST and should return the new symbolic expression for the symbolic state."""
    if not expr is None and expr[0] in BINARY_OPS:
        return evaluate_expr_to_smt(expr[1], expr[2], op_map[expr[0]], s, m)


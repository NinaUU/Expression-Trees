import math

# split a string into mathematical tokens
# returns a list of numbers, operators, parantheses and commas
# output will not contain spaces
def tokenize(string):
    splitchars = list("+-*/(),")
    
    # surround any splitchar by spaces
    tokenstring = []
    for c in string:
        if c in splitchars:
            tokenstring.append(' %s ' % c)
        else:
            tokenstring.append(c)
    tokenstring = ''.join(tokenstring)
    #split on spaces - this gives us our tokens
    tokens = tokenstring.split()
    
    #special casing for **:
    ans = []
    for t in tokens:
        if len(ans) > 0 and t == ans[-1] == '*':
            ans[-1] = '**'
        else:
            ans.append(t)
    return ans
    
# check if a string represents a numeric value
def isnumber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# check if a string represents an integer value        
def isint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

class Expression():
    """A mathematical expression, represented as an expression tree"""
    
    """
    Any concrete subclass of Expression should have these methods:
     - __str__(): return a string representation of the Expression.
     - __eq__(other): tree-equality, check if other represents the same expression tree.
    """
    # TODO: when adding new methods that should be supported by all subclasses, add them to this list
    
    # operator overloading:
    # this allows us to perform 'arithmetic' with expressions, and obtain another expression
    def __add__(self, other):
        return AddNode(self, other)
    
    def __sub__(self,other):
        return SubNode(self,other)

    def __mul__(self,other):
        return MulNode(self,other)
    
    def __div__(self,other):
        return DivNode(self,other)
    
    def __pow__(self,other):
        return PowNode(self,other)
        
    # TODO: other overloads, such as __sub__, __mul__, etc.
    
    # basic Shunting-yard algorithm
    def fromString(string): # van een string (infix) naar expressieboom
        # split into tokens
        tokens = tokenize(string)

        # stack used by the Shunting-Yard algorithm
        stack = []
        # output of the algorithm: a list representing the formula in RPN
        # this will contain Constant's and '+'s
        output = []

        ## rang van operators
        first_op_list = ['**']
        second_op_list = ['*','/']
        trird_op_list = ['+','-']
        haakje_list = ['(',')']
        # list of operators
        oplist = ['+', '-']+second_op_list+first_op_list+['(',')']

        for token in tokens:
            if isnumber(token):
                # numbers go directly to the output
                if isint(token):
                    output.append(Constant(int(token)))
                else:
                    output.append(Constant(float(token)))
            elif token in oplist:
                # pop operators from the stack to the output until the top is no longer an operator
                while True:
                    if token in second_op_list and stack[-1] in trird_op_list:
                        break ## dan moet hij op de stack
                    if token in first_op_list and stack[-1] in second_op_list:
                        break
                    if token in first_op_list and stack[-1] in third_op_list:
                        break ##alle variaties van lagere rang
                    if token in third_op_list and stack[-1] in third_op_list:
                        break ## machtsverheven in rechtsacciosatief, dus moeten achter elkaar op de stack

                    # TODO: when there are more operators, the rules are more complicated
                    # look up the shunting yard-algorithm
                    ## werkt nu voor plus en min, allecombinaties
                    if len(stack) == 0 or stack[-1] not in oplist:
                        break
                    output.append(stack.pop())
                # push the new operator onto the stack
                stack.append(token)
            elif token == '(':
                # left parantheses go to the stack
                stack.append(token)
            elif token == ')':
                # right paranthesis: pop everything upto the last left paranthesis to the output
                while not stack[-1] == '(':
                    output.append(stack.pop())
                # pop the left paranthesis from the stack (but not to the output)
                stack.pop()
            # TODO: do we need more kinds of tokens?
            else:
                # unknown token
                raise ValueError('Unknown token: %s' % token)

        # pop any tokens still on the stack to the output
        while len(stack) > 0:
            output.append(stack.pop())

        # convert RPN to an actual expression tree
        for t in output:
            if t in oplist:
                # let eval and operator overloading take care of figuring out what to do
                y = stack.pop()
                x = stack.pop()
                stack.append(eval('x %s y' % t)) ## combineer node met x en node met y met ouder node operator
            else:
                # a constant, push it to the stack
                stack.append(t)
        # the resulting expression tree is what's left on the stack
        return stack[0]
    
class Constant(Expression):
    """Represents a constant value"""
    def __init__(self, value):
        self.value = value
        
    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        else:
            return False
        
    def __str__(self):
        return str(self.value)
        
    # allow conversion to numerical values
    def __int__(self):
        return int(self.value)
        
    def __float__(self):
        return float(self.value)
        
class Variables(Expression):
    """Reprecenteerd een variabele"""
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return str(self.value)

class BinaryNode(Expression):
    """A node in the expression tree representing a binary operator."""
    
    def __init__(self, lhs, rhs, op_symbol):
        self.lhs = lhs
        self.rhs = rhs
        self.op_symbol = op_symbol
    
    # TODO: what other properties could you need? Precedence, associativity, identity, etc.
            
    def __eq__(self, other):
        if type(self) == type(other):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False
            
    def __str__(self):
        lstring = str(self.lhs)
        rstring = str(self.rhs)
        
        # TODO: do we always need parantheses?
        return "(%s %s %s)" % (lstring, self.op_symbol, rstring)
        
class AddNode(BinaryNode):
    """Represents the addition operator"""
    def __init__(self, lhs, rhs):
        super(AddNode, self).__init__(lhs, rhs, '+')

class SubNode(BinaryNode):
    """Represents the subtraction operator"""
    def __init__(self,lhs,rhs):
        super(SubNode,self).__init__(lhs,rhs,'-')

class MulNode(BinaryNode):
    """Represents the multiplication operator"""
    def __init__(self,lhs,rhs):
        super(MulNode,self).__init__(lhs,rhs,'*')

class DivNode(BinaryNode):
    """Represents the division operator"""
    def __init__(self,lhs,rhs):
        super(DivNode,self).__init__(lhs,rhs,'/')      
        
class PowNode(BinaryNode):
    """Represents the power operator"""
    def __init__(self,lhs,rhs):
        super(PowNode,self).__init__(lhs,rhs,'**')        
# TODO: add more subclasses of Expression to represent operators, variables, functions, etc.

#test

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

    #special casing for negative numbers:
    if ans[0]=='-':
        a=ans.pop(0)
        b=ans.pop(0) # 0th element was previously 1st element but since element 0 has been removed, element 1 has become the new 0th element
        ans=[str(a)+str(b)]+ans 
    for i in range(1,len(ans)-1):
        if ans[i]=='-' and ans[i-1] in splitchars:
            a=ans.pop(i)
            b=ans.pop(i) # ith element was previously (i+1)th element
            ans.insert(i,str(a)+str(b))

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

# returns the number of priority of an operator
# the lower the priority, the higher the number
def op_nummer(op):
    if op == '**':
        return 1
    if op == '*' or op == '/':
        return 2
    if op == '+' or op == '-':
        return 3

# returns priority number of the operator in the string with lowest priority
# the lower the priority, the higher the number
def zoek_op(st):
    result = 1
    for s in st:
        if (s == '*' or s == '/') and result != 3:
            result = 2
        if s == '+' or s == '-':
            result = 3
    return result

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
    

    def __truediv__(self,other):

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
        third_op_list = ['-']
        fourth_op_list=['+']
        # list of operators
        oplist = third_op_list+second_op_list+first_op_list+fourth_op_list


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
                    if len(stack) == 0 or stack[-1] not in oplist:
                        break
                    if token in second_op_list and stack[-1] in third_op_list+fourth_op_list:
                        break ## dan moet hij op de stack
                    if token in first_op_list and stack[-1] in second_op_list+third_op_list+fourth_op_list:
                        break
                    if token in third_op_list and stack[-1] in fourth_op_list:
                        break
                    if token in fourth_op_list and stack[-1] in fourth_op_list:
                        break 
                    output.append(stack.pop())
                    
                    if token in second_op_list and stack[-1] in trird_op_list:
                        break ## dan moet hij op de stack
                    if token in first_op_list and stack[-1] in second_op_list:
                        break
                    if token in first_op_list and stack[-1] in third_op_list:
                        break ##alle variaties van lagere rang
                    if token in third_op_list and stack[-1] in third_op_list:
                        break ## machtsverheven in rechtsacciosatief, dus moeten achter elkaar op de stack

                    # TODO: when there are more operators, the rules are more complicated
                    ## werkt nu voor plus en min, allecombinaties
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
            elif isinstance(token, str):
                output.append(Variables(token)) 
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

    def __eq__(self,other):
        op_commutative=['+','*'] # list of commutative operators
        n_op_commutative=['-','/','**'] # list of operators that are not commutative
        if isinstance(self,BinaryNode) and isinstance(other,BinaryNode): # recursive code since there are more nodes below that also need to be compared
            if self.op_symbol!=other.op_symbol:
                return False
            # from here on we know self.op_symbol==other.op_symbol
            if self.op_symbol in n_op_commutative: 
                return self.lhs.__eq__(other.lhs) and self.rhs.__eq__(other.rhs)
            elif self.op_symbol in op_commutative: 
                return (self.lhs.__eq__(other.lhs) and self.rhs.__eq__(other.rhs)) or (self.lhs.__eq__(other.rhs) and self.rhs.__eq__(other.lhs))
        elif isinstance(self,Constant) and isinstance(other,Constant): # only one execution since Constants only occur in leaves
            return self.value==other.value
        else: # if the types are not the same, the nodes cannot be compared
            return False

    def evaluate(self,dictionary={}):
        """ A function that calculates the numerical value of an expression.
            dictionary = a dictionary assigning values to the variables """
        if isinstance(self.lhs,BinaryNode) or isinstance(self.rhs,BinaryNode): # recursive loop until no more operators are encountered
            if isinstance(self.lhs,BinaryNode) and isinstance(self.rhs,BinaryNode):
                return eval(str(self.lhs.evaluate(dictionary))+self.op_symbol+str(self.rhs.evaluate(dictionary)))
            if isinstance(self.lhs,BinaryNode):
                return eval(str(self.lhs.evaluate(dictionary))+self.op_symbol+str(self.rhs.value))
            if isinstance(self.rhs,BinaryNode):
                return eval(str(self.lhs.value)+self.op_symbol+str(self.rhs.evaluate(dictionary)))
        # no more operators encountered means that the next nodes are constants or variables or a combination
        return eval(str(self.lhs.value)+self.op_symbol+str(self.rhs.value),dictionary)
            
        
class Constant(Expression):
    """Represents a constant value"""
    def __init__(self, value):
        self.value = value
        
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
    
    def __eq__(self,other):
        if isinstance(other, Variables):
            return self.value == other.value
        else:
            return False
        
class Variables(Expression):
    """Reprecenteerd een variabele"""
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return str(self.value)
    
    def __eq__(self,other):
        if isinstance(other, Variables):
            return self.value == other.value
        else:
            return False
        
class BinaryNode(Expression):
    """A node in the expression tree representing a binary operator."""
    
    def __init__(self, lhs, rhs, op_symbol):
        self.lhs = lhs
        self.rhs = rhs
        self.op_symbol = op_symbol
    
    # TODO: what other properties could you need? Precedence, associativity, identity, etc.
    def __str__(self):
        lstring = str(self.lhs)
        rstring = str(self.rhs)

        # Haakjes zijn alleen nodig wanneer er in de uitdrukking links of rechts van de operator een
        # andere operator voorkomt die een lagere prioriteit heeft.
        # Onderstaande code checkt of dit het geval is
        if zoek_op(lstring) > op_nummer(self.op_symbol) and zoek_op(rstring) <= op_nummer(self.op_symbol):
            return "(%s) %s %s" % (lstring, self.op_symbol, rstring)
        if zoek_op(lstring) <= op_nummer(self.op_symbol) and zoek_op(rstring) > op_nummer(self.op_symbol):
            return "%s %s (%s)" % (lstring, self.op_symbol, rstring)
        if zoek_op(lstring) > op_nummer(self.op_symbol) and zoek_op(rstring) > op_nummer(self.op_symbol):
            return "(%s) %s (%s)" % (lstring, self.op_symbol, rstring)
        return "%s %s %s" % (lstring, self.op_symbol, rstring)
        
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


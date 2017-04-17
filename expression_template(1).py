import math

def tokenize(string):
    """ A function that splits a string into mathematical tokens. Returns a list of numbers, operators, parantheses and commas.
        Output will not contain spaces. """
    splitchars = list("+-*/(),")

    # surround any splitchar by spaces
    tokenstring = []
    for c in string:
        if c in splitchars:
            tokenstring.append(' %s ' % c)
        else:
            tokenstring.append(c)
    tokenstring = ''.join(tokenstring)
    # split on spaces - this gives us our tokens
    tokens = tokenstring.split()

    # special casing for **:
    ans = []
    for t in tokens:
        if len(ans) > 0 and t == ans[-1] == '*':
            ans[-1] = '**'
        else:
            ans.append(t)

    # special casing for negative numbers:
    # if input string starts with a minus (special casing for **):
    if ans[0] == '-':
        if len(ans) == 2:
            a = ans.pop(0) # remove the first two elements of ans
            b = ans.pop(0)  
            ans = [str(a) + str(b)] + ans # merge the two removed elements into one
        elif len(ans) > 2:
            if ans[2] != '**':
                a = ans.pop(0)
                b = ans.pop(0)
                ans = [str(a) + str(b)] + ans
            else:
                ans = [0] + ans
    # if minus sign is encountered right after another operator (special casing for **):
    for i in range(1, len(ans) - 1):
        if ans[i] == '-' and ans[i - 1] in ['+', '-', '*', '/', '**', '(']:
            if len(ans) <= i + 2:
                a = ans.pop(i)  
                b = ans.pop(i)  
                ans.insert(i, str(a) + str(b))
            elif len(ans) > i + 2:
                if ans[i + 2] != '**':
                    a = ans.pop(i)
                    b = ans.pop(i)
                    ans.insert(i, str(a) + str(b))
    return ans


# check if a string represents a numeric value:
def isnumber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# check if a string represents an integer value:
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


# returns the priority of the operator with the lowest priority
# (highest number) that is not yet surrounded by brackets
def zoek_op(st):
    open = 0
    result = 1
    for s in st:
        if (s == '*' or s == '/') and (result != 3) and (open == 0):
            result = 2
        if (s == '+' or s == '-') and open == 0:
            result = 3
        if s == '(':
            open += 1
        if s == ')':
            open -= 1
    return result


class Expression():
    """ A mathematical expression, represented as an expression tree. """

    # operator overloading:
    # this allows us to perform 'arithmetic' with expressions, and obtain
    # another expression
    def __add__(self, other):
        return AddNode(self, other)

    def __sub__(self, other):
        return SubNode(self, other)

    def __mul__(self, other):
        return MulNode(self, other)

    def __rmul__(self, other):
        return MulNode(self, other)

    def __truediv__(self, other):

        return DivNode(self, other)

    def __pow__(self, other):
        return PowNode(self, other)

    def log(self):
        return LogNode(self)

    def sin(self):
        return SinNode(self)

    def cos(self):
        return CosNode(self)

    def tan(self):
        return TanNode(self)

    def fromString(string):
        """ A function that takes a string as input and returns an expression tree, implementing the Shunting-yard algorithm. """
        # split into tokens
        tokens = tokenize(string)
        # stack used by the Shunting-yard algorithm
        stack = []
        # output of the algorithm: a list representing the formula in RPN
        # this will contain Constants, Variables, operators, functions and parantheses
        output = []
        # list of operators, ordered from lowest to highest priority
        oplist = ['+', '-', '*', '/', '**']
        precedence = {'+': 0, '-': 1, '*': 2, '/': 2, '**': 3}
        functionlist = ['log', 'sin', 'tan', 'cos']

        for token in tokens:
            if isnumber(token):
                # numbers go directly to the output
                if isint(token):
                    output.append(Constant(int(token)))
                else:
                    output.append(Constant(float(token)))
            elif token in oplist:
                # pop operators from the stack to the output until the top is
                # no longer an operator
                while True:
                    if len(stack) == 0 or stack[-1] not in oplist:
                        break
                    if precedence[token] > precedence[stack[-1]]:
                        break
                    output.append(stack.pop())
                # push the new operator onto the stack
                stack.append(token)
            elif token in functionlist:
                stack.append(token)
            elif token == '(':
                # left parantheses go to the stack
                stack.append(token)
            elif token == ')':
                # right paranthesis: pop everything upto the last left
                # paranthesis to the output
                while not stack[-1] == '(':
                    output.append(stack.pop())
                # pop the left paranthesis from the stack (but not to the
                # output)
                stack.pop()
                if stack != [] and stack[-1] in functionlist:
                    output.append(stack.pop())
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
                # let eval and operator overloading take care of figuring out
                # what to do
                y = stack.pop()
                x = stack.pop()
                # combine node containing variable x and node containing variable y with parent node (operator)
                stack.append(eval('x %s y' % t))
            elif t in functionlist:
                x = stack.pop()
                stack.append(eval('Expression.%s(x)' % t))
            else:
                # a constant or variable, push it to the stack
                stack.append(t)
        # the resulting expression tree is what's left on the stack
        return stack[0]

    
    def __eq__(self, other): 
        """ Overload of the equality operator, ==. Checks if two expression trees are equal.
            Includes casing for the subclasses, which no longer have their own overload functions. """
        op_commutative = ['+', '*']  # list of commutative operators
        n_op_commutative = ['-', '/', '**']  # list of operators that are not commutative
        # recursive code since there are more nodes below that also need to be
        # compared
        if isinstance(self, BinaryNode) and isinstance(other, BinaryNode):
            if self.op_symbol != other.op_symbol:
                return False
            # from here on we know self.op_symbol==other.op_symbol
            if self.op_symbol in n_op_commutative:
                return self.lhs.__eq__(other.lhs) and self.rhs.__eq__(other.rhs)
            elif self.op_symbol in op_commutative:
                return (self.lhs.__eq__(other.lhs) and self.rhs.__eq__(other.rhs)) or (self.lhs.__eq__(other.rhs) and self.rhs.__eq__(other.lhs))
        elif isinstance(self, MonoNode) and isinstance(other, MonoNode):
            return type(self) == type(other) and self.lhs.__eq__(other.lhs)
        # only one execution since Constants only occur in leaves:
        elif isinstance(self, Constant) and isinstance(other, Constant):
            return self.value == other.value
        elif isinstance(self, Variables) and isinstance(other, Variables):
            return self.value == other.value
        else:  # if the types are not the same, the nodes cannot be compared
            return False
        
    def evaluate(self, dictionary={}):
        """ A function that calculates the numerical value of an expression.
            dictionary is a dictionary assigning values to the variables, if not specified: empty dictionary. """
        if isinstance(self, BinaryNode):  # recursive loop until no more operators are encountered
            return eval(str(self.lhs.evaluate(dictionary)) + self.op_symbol + str(self.rhs.evaluate(dictionary)))
        # no more operators encountered means that the next nodes are constants or variables or a combination
        return eval(str(self.value), dictionary)

    def part_evaluate(self, dictionary={}):
        """ A function that can partially evaluate an expression. 
            dictionary = a dictionary that may assign values to none, some or all of the variables. """
        newstring = '' # string in which new expression is saved
        if isinstance(self, BinaryNode): # recursive code for walking down the tree
            newstring += str(self.lhs.part_evaluate(dictionary)) + self.op_symbol + str(
                self.rhs.part_evaluate(dictionary))
        if isinstance(self, Variables):
            if str(self.value) not in dictionary:
                newstring += self.value # leave variable unchanged
            else:
                newstring += str(eval(self.value, dictionary)) # substitute value for variable
        if isinstance(self, Constant):
            newstring += str(self.value)
        if isinstance(self, MonoNode): # recursive code for evaluating the argument/input of the function
            newstring += self.op_symbol + '(' + str(self.lhs.part_evaluate(dictionary)) + ')'
        tree = Expression.fromString(newstring) # make a new tree that is partially evaluated
        try:
            return tree.evaluate() # simplify if possible
        except NameError:
            return tree

    def __neg__(self):
        """ A function that returns the negative of an expression tree and overloads the - that is placed in front of an expression tree """
        return Expression.fromString('(-1)*(' + str(self) + ')')

    def simplify(self):
        """ A function that simplifies a function by using standard rules.
        It modifies the expression, so later on the simplified expression is used."""
        if isinstance(self, BinaryNode):
            self.lhs = self.lhs.simplify()
            self.rhs = self.rhs.simplify()
            if isinstance(self, MulNode):
                if isinstance(self.lhs, Constant) and isinstance(self.rhs, Constant):
                    self = Constant(self.lhs.value * self.rhs.value)
                elif self.lhs == Constant(0) or self.rhs == Constant(0):
                    self = Constant(int(0))
                elif self.lhs == Constant(1):
                    self = self.rhs
                elif self.rhs == Constant(1):
                    self = self.lhs
            elif isinstance(self, AddNode):
                if isinstance(self.lhs, Constant) and isinstance(self.rhs, Constant):
                    self = Constant(self.lhs.value + self.rhs.value)
                elif self.lhs == Constant(0):
                    self = self.rhs
                elif self.rhs == Constant(0):
                    self = self.lhs
            elif isinstance(self, SubNode):
                if isinstance(self.lhs, Constant) and isinstance(self.rhs, Constant):
                    self = Constant(self.lhs.value - self.rhs.value)
                elif self.lhs == Constant(0):
                    self = -self.rhs
                elif self.rhs == Constant(0):
                    self = self.lhs
            elif isinstance(self, MonoNode):
                self.lhs = self.lhs.simplify()
                if isinstance(self, LogNode):
                    if self.lhs == Constant(1):
                        self = Constant(0)
                elif isinstance(self, SinNode):
                    if self.lhs == Constant(0):
                        self = Constant(0)
                elif isinstance(self, CosNode):
                    if self.lhs == Constant(0):
                        self = Constant(1)
            elif isinstance(self, Constant) or isinstance(self, Variables):
                return self  # end of tree
        return self

    def derivative(self, x): 
        """ A function that returns the derivative of an expession with respect to x """
        if isinstance(self, Constant):  # The derivative of a constant is 0
            return Constant(0)
        if isinstance(self, Variables):
            if self == Variables(x):  # The derivative of x is 1
                return Constant(1)
            else:
                return Constant(0)  # The derivative of other variables is 0
        if isinstance(self, SinNode):
            return CosNode(self.lhs)*self.lhs.derivative(x)
        if isinstance(self, CosNode):
            return -SinNode(self.lhs)*self.lhs.derivative(x)
        if isinstance(self,TanNode):
            return ((TanNode(self.lhs))**2 + 1)*self.lhs.derivative(x)
        if isinstance(self,LogNode):
            return self.lhs.derivative(x)/self.lhs
        if self.op_symbol == '+':  # The derivative of a sum is the sum of the derivatives
            return self.lhs.derivative(x) + self.rhs.derivative(x)
        if self.op_symbol == '-':  # The derivative of a difference is the difference of the derivatives
            return self.lhs.derivative(x) - self.rhs.derivative(x)
        if self.op_symbol == '*':  # The product rule
            return self.lhs * self.rhs.derivative(x) + self.lhs.derivative(x) * self.rhs
        if self.op_symbol == '/':  # The quotient rule
            return (self.rhs * self.lhs.derivative(x) - self.lhs * self.rhs.derivative(x)) / (self.rhs ** Constant(2))
        # The power rule
        if self.op_symbol == '**' and isinstance(self.rhs, Constant):
            return self.rhs * self.lhs ** (self.rhs.value - 1) * self.lhs.derivative(x)
        if self.op_symbol == '**': #Logarithmic differentiation
            return self * (self.rhs.derivative(x)*LogNode(self.lhs) +
                    self.lhs.derivative(x)*self.rhs/self.lhs)

    def solve(self, x, y=0, x0=0, tolerance=10 ** (-5), n=1000):
        """A function that finds the root of an expression by using the Newton algorithm.
           Input x should be a string of the variable for which the expression should be solved.
           To prevent endless loops, a maximum number of iterations has been set. """
        div = Expression.derivative(self, x).simplify()
        xm = x0 - (self.evaluate({x: x0})) / (div.evaluate({x: x0}))
        if n != None:
            k = 2
            for k in range(2, n):
                ym = self.evaluate({x: xm})
                xn = xm - ym / (div.evaluate({x: xm}))
                xm = xn
                if abs(ym) < tolerance:
                    return xm
        else:
            while True:
                ym = self.evaluate({x: xm})
                xn = xm - ym / (div.evaluate({x: xm}))
                xm = xn
                if abs(ym) < tolerance:
                    return xm
        raise ValueError('Did not find a root of %s within %s itterations whit x0 = %s and tolerance = %s' % self, n,
                         x0, tolerance)
    
    def num_integration(self, a, b, x, n,dict={}):  
        """ A function that approximates the integral of the expression with respect to x between a and b with a Riemann sum
            n is the number of pieces into which the interval is divided """
        s = self.part_evaluate(dict)
        n = n  
        xi = a
        dx = (b - a) / (n)
        result = 0
        for i in range(n):
            result += s.evaluate({x: xi}) * dx
            xi += dx
        return result


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
    """Represents a variable"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class BinaryNode(Expression):
    """A node in the expression tree representing a binary operator."""

    def __init__(self, lhs, rhs, op_symbol):
        self.lhs = lhs
        self.rhs = rhs
        self.op_symbol = op_symbol

    def __str__(self):
        lstring = str(self.lhs)
        rstring = str(self.rhs)
        
        # Parantheses are only needed when an operator of lower priority occurs to the left or to the right of an operator
        # Checks whether this is the case
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

    def __init__(self, lhs, rhs):
        super(SubNode, self).__init__(lhs, rhs, '-')


class MulNode(BinaryNode):
    """Represents the multiplication operator"""

    def __init__(self, lhs, rhs):
        super(MulNode, self).__init__(lhs, rhs, '*')


class DivNode(BinaryNode):
    """Represents the division operator"""

    def __init__(self, lhs, rhs):
        super(DivNode, self).__init__(lhs, rhs, '/')


class PowNode(BinaryNode):
    """Represents the power operator"""

    def __init__(self, lhs, rhs):
        super(PowNode, self).__init__(lhs, rhs, '**')


class MonoNode(Expression):
    """ A node for functions that request a single argument/input, such as sin or log. """

    def __init__(self, lhs, op_symbol):
        self.lhs = lhs
        self.op_symbol = op_symbol

    def __str__(self):
        lstring = str(self.lhs)
        return "%s(%s)" % (self.op_symbol, lstring)


class LogNode(MonoNode):
    """Represents the logarithmic function (base e)"""

    def __init__(self, lhs):
        super(LogNode, self).__init__(lhs, "log")

    def evaluate(self, dictionary={}):
        value = eval(str(self.lhs), dictionary)
        return math.log(value)


class SinNode(MonoNode):
    """Represents the sine function"""

    def __init__(self, lhs):
        super(SinNode, self).__init__(lhs, "sin")

    def evaluate(self, dictionary={}):
        value = eval(str(self.lhs), dictionary)
        return math.sin(value)


class CosNode(MonoNode):
    """Represents the cosine function"""

    def __init__(self, lhs):
        super(CosNode, self).__init__(lhs, "cos")

    def evaluate(self, dictionary={}):
        value = eval(str(self.lhs), dictionary)
        return math.cos(value)


class TanNode(MonoNode):
    """Represents the tangent function"""

    def __init__(self, lhs):
        super(TanNode, self).__init__(lhs, "tan")

    def evaluate(self, dictionary={}):
        value = eval(str(self.lhs), dictionary)
        return math.tan(value)


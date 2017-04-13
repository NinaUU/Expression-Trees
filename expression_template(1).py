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
    if ans[0]=='-':
        if len(ans)==2:
            a=ans.pop(0)
            b=ans.pop(0) # 0th element was previously 1st element but since element 0 has been removed, element 1 has become the new 0th element
            ans=[str(a)+str(b)]+ans
        elif len(ans)>2:
            if ans[2]!='**':
                a=ans.pop(0)
                b=ans.pop(0)
                ans=[str(a)+str(b)]+ans
            else:
                ans=[0]+ans
    # if minus sign is encountered straight after another operator (special casing for **):
    for i in range(1,len(ans)-1):
        if ans[i]=='-' and ans[i-1] in ['+','-','*','/','**','(']:
            if len(ans)<=i+2:
                a=ans.pop(i)
                b=ans.pop(i) # ith element was previously (i+1)th element
                ans.insert(i,str(a)+str(b))
            elif len(ans)>i+2:
                if ans[i+2]!='**':
                    a=ans.pop(i)
                    b=ans.pop(i)
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


#geeft de prioriteit terug van de operator met laagste priorieit
# (dus hoogste getal) die nog niet ingesloten in door haakjes
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
    """A mathematical expression, represented as an expression tree"""

    """
    Any concrete subclass of Expression should have these methods:
     - __str__(): return a string representation of the Expression.
     - __eq__(other): tree-equality, check if other represents the same expression tree.
    """
    # TODO: when adding new methods that should be supported by all
    # subclasses, add them to this list

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

    # basic Shunting-yard algorithm
    def fromString(string):  # van een string (infix) naar expressieboom
        # split into tokens
        tokens = tokenize(string)

        # stack used by the Shunting-Yard algorithm
        stack = []
        # output of the algorithm: a list representing the formula in RPN
        # this will contain Constant's and '+'s
        output = []
        # list of operators, op volgoder van miste naar meeste voorrang
        oplist = ['+', '-', '*', '/', '**']
        precidence = {'+': 0, '-': 1, '*': 2, '/': 2, '**': 3}
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
                    if precidence[token] > precidence[stack[-1]]:
                        break
                    output.append(stack.pop())
                # push the new operator onto the stack
                stack.append(token)
            elif token in functionlist:
                # TODO: wat is de voorrang van log, Sin, ect
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
                # combineer node met x en node met y met ouder node operator
                stack.append(eval('x %s y' % t))
            elif t in functionlist:
                x = stack.pop()
                stack.append(eval('Expression.%s(x)' % t))
            else:
                # a constant, push it to the stack ## of variabele
                stack.append(t)
        # the resulting expression tree is what's left on the stack
        return stack[0]

    def __eq__(self, other): # TODO: MonoNode
        op_commutative = ['+', '*']  # list of commutative operators
        # list of operators that are not commutative
        n_op_commutative = ['-', '/', '**']
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
        # only one execution since Constants only occur in leaves
        elif isinstance(self, Constant) and isinstance(other, Constant):
            return self.value == other.value
        else:  # if the types are not the same, the nodes cannot be compared
            return False

    def evaluate(self,dictionary={}): # TODO: log, sin, cos, tan
        """ A function that calculates the numerical value of an expression.
            dictionary = a dictionary assigning values to the variables """
        if isinstance(self,BinaryNode): # recursive loop until no more operators are encountered
            return eval(str(self.lhs.evaluate(dictionary))+self.op_symbol+str(self.rhs.evaluate(dictionary)))
        # no more operators encountered means that the next nodes are constants or variables or a combination
        return eval(str(self.value),dictionary)

    def __neg__(self):
        return Expression.fromString('(-1)*('+str(self)+')')

    def symplify(self):
        """ A function that simplifies a function by using standard rules.
        It modefies the expression, so later on te symplified expression is used."""
        # TODO: a(b+c)= ab + ac, sin2 + con2 = 1,
        # ax+bx = (a+b)x,
        if isinstance(self, BinaryNode):
            self.lhs = self.lhs.symplify()
            self.rhs = self.rhs.symplify()
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
                    self = Constant(self.lhs.value + self.rhs.value)
                elif self.lhs == Constant(0):
                    self = -self.rhs
                elif self.rhs == Constant(0):
                    self = self.lhs
            elif isinstance(self, MonoNode):
                self.lhs = self.lhs.symplify()
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
                return self  # einde van de boom
        return self

    def derivative(self, x):  # returns the derivative of the expression with respect to x
        if isinstance(self, Constant):  # The derivative of a constant is 0
            return Constant(0)
        if isinstance(self, Variables):
            if self == Variables(x):  # The derivative of x is 1
                return Constant(1)
            else:
                return Constant(0)  # The derivative of other variables is 0
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
        # if self.op_symbol == '**': #Logaritmisch differentiëren(werkt nog niet want log() bestaat nog niet)
            # return self * (self.rhs.derivative(x)*log(self.lhs) +
            # self.lhs.derivative(x)*self.rhs/self.lhs)

    def solve(self, x, y=0, x0=0, toletance = 10**(-5), n=1000):
        """A function that finds the root of a Expression by using the Netwon alogaritm.
        To prefent endless loops, the maximal itterations has been set"""
        div=Expression.derivative(self, x).symplify()
        xm=x0 - (self.evaluate({x: x0})) / (div.evaluate({x: x0}))
        if n != None:
            k = 2
            for k in range(2, n):
                ym = self.evaluate({x: xm})
                xn=xm - ym / (div.evaluate({x: xm}))
                xm=xn
                if abs(ym) < toletance:
                    return xm
        else:
            while True:
                ym = self.evaluate({x: xm})
                xn=xm - ym / (div.evaluate({x: xm}))
                xm=xn
                if abs(ym) < toletance:
                    return xm
        raise ValueError('Did not find a root of %s within %s itterations whit x0 = %s and toletance = %s' % self, n ,x0, toletance)

    def num_integration(self,a,b,x,n): #approximates the integral of the expression with resprect to x between a and b with a riemann sum
        n = n #number of steps
        xi = a
        dx = (b-a)/(n)
        result = 0
        for i in range(n):
            result += self.evaluate({x:xi})*dx
            xi += dx
        return result

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

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
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

    # TODO: what other properties could you need? Precedence, associativity,
    # identity, etc.

    def __eq__(self, other):
        if type(self) == type(other):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False

    def __str__(self):
        lstring = str(self.lhs)
        rstring = str(self.rhs)

        # TODO: do we always need parantheses?
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
    """ Een node variand voor functies die een variabele vragen, zoals sin of log."""

    def __init__(self, lhs, op_symbol):
        self.lhs = lhs
        self.op_symbol = op_symbol

    def __eq__(self, other):
        if type(self) == type(other):
            return self.lhs == other.lhs
        else:
            return False

    def __str__(self):
        lstring = str(self.lhs)
        return "%s(%s)" % (self.op_symbol, lstring)

        # TODO Haakjes weg werken?


class LogNode(MonoNode):
    """Representeert de log functie"""

    def __init__(self, lhs):
        super(LogNode, self).__init__(lhs, "log")

    def evaluate(self,dictionary={}):
        value=eval(str(self.lhs),dictionary)
        return math.log(value)

class SinNode(MonoNode):
    """Representeert de sinus functie"""

    def __init__(self, lhs):
        super(SinNode, self).__init__(lhs, "sin")

    def evaluate(self,dictionary={}):
        value=eval(str(self.lhs),dictionary)
        return math.sin(value)

class CosNode(MonoNode):
    """Representeert de cosinus functie"""

    def __init__(self, lhs):
        super(CosNode, self).__init__(lhs, "cos")

    def evaluate(self,dictionary={}):
        value=eval(str(self.lhs),dictionary)
        return math.cos(value)

class TanNode(MonoNode):
    """Representeert de tangus functie"""

    def __init__(self, lhs):
        super(TanNode, self).__init__(lhs, "tan")

    def evaluate(self,dictionary={}):
        value=eval(str(self.lhs),dictionary)
        return math.tan(value)

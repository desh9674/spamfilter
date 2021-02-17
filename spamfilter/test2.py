import decimal

y = "5.73rf"

if type(eval(y)) == float:
    print(y)

x = decimal.Decimal("0.123")
print(x)
print(0.0<x<1)

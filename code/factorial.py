# -*- coding: utf-8 -*-
while(1):
    num = int(input('請輸入階層數字:'))
    
    if num == 0:
        print(1)
    elif num < 0:
        print('number should be positive interger')
    else:
        stack = [i for i in range(1, num + 1)]
        for i in range(len(stack[:]) - 1):
            F = stack.pop()
            A = stack.pop()
            mul = F * A
            stack.append(mul)
        print(stack[0])

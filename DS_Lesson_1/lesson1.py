import random as r

num = input('Введите номер задачи:')

while num != 'exit':
    if num == '1':
        print('1.  Создать кортеж, содержащий 4 разных числа. Вывести на экран значение второго элемента кортежа.')
        nums = (r.randint(1, 5), r.randint(1, 5),  r.randint(1, 5), r.randint(1, 5))
        print(nums)
        print(nums[1])
    if num == '2':
        print('2.  Написать программу, которая принимает строку от пользователя и выводит количество символов в этой строке.')
        print(len(input('Введите строку:')))
    if num == '3':
        print('3.  Создать словарь, содержащий информацию о студенте: имя, возраст, курс. Вывести всю информацию о студенте на экран.')
        student = {'Name': 'Андрей', 'Age': '174', 'course': 'python разработчик'}

        print('Имя', student['Name'])
        print('Возраст', student['Age'])
        print('Курс', student['course'])
    if num == '4':
        print('4.  Используя цикл, создать список, содержащий все целые числа от 1 до 10, и вывести его на экран.')

        spisok = []
        for i in range(10):
            spisok.append(i + 1)

        print(spisok)
    if num == '5':
        print('5.  Написать функцию, которая принимает список чисел и возвращает их сумму.')
        nums = input('Введите числа через пробел').split(' ')
        sum = 0
        for n in nums:
            sum += int(n)

        print(sum)

    if num == '6':
        print('6.  Создать множество из 5 разных чисел, затем добавить в него новое число и вывести на экран.')

        m = {1, 2, 3, 4, 5}
        print(m)
        m.add(6)
        print(m)

    if num == '7':
        print('7.  Написать программу, которая принимает от пользователя целое число и выводит на экран его квадрат')

        number = int(input('Введите число:'))

        print(number * number)

    if num == '8':
        print('8.  Создать словарь с пятью парами «ключ-значение», где ключи - названия фруктов, а значения - их цвета. Вывести значения всех ключей')

        fruits = {'apple': 'red', 'banana': 'yellow', 'pineapple': 'yellow', 'orange': 'orange', 'blueberry': 'blue'}

        for k in fruits:
            print(k, fruits[k])

    if num == '9':
        print('9.  Написать функцию, которая принимает строку и выводит ее в обратном порядке.')

        string = input('Введите строку:')
        
        spisok = string.split(' ')
        spisok.reverse()
        string2 = ' '.join(spisok)
        print(string2)
    
    if num == '10':
        print('10.  Создать список из 5 строк и заменить в нем третий элемент на новую строку')

        spisok = ['hello', 'world', 'meow', 'cat', 'dog']

        new = input('Введите новую строку:')

        spisok[2] = new
        print(spisok)

    num = input('Введите номер задачи:')
import random as r

num = input('Введите номер задачи (выход - exit):')

while num != 'exit':
    if num == '1':
        print('1.  Создать кортеж, содержащий 4 разных числа. Вывести на экран значение второго элемента кортежа.')
        nums = (r.randint(1, 5), r.randint(1, 5),  r.randint(1, 5), r.randint(1, 5))
        print("Второй элемент кортежа:", nums[1])
        print("Сложность: O(1) - доступ к элементу кортежа")

    if num == '2':
        print('2.  Написать программу, которая принимает строку от пользователя и выводит количество символов в этой строке.')
        print("Количество символов в строке:", len(input('Введите строку:')))
        print("Сложность этой задачи является константной (O(1))." 
              + " Потому что функция len() в Python работает за постоянное время вне зависимости от длины строки."
               + " Она просто возвращает заранее вычисленное значение длины строки.")
        
    if num == '3':
        print('3.  Создать словарь, содержащий информацию о студенте: имя, возраст, курс. Вывести всю информацию о студенте на экран.')
        student = {'Name': 'Андрей', 'Age': '174', 'course': 'python разработчик'}
        print(f' Имя: {student["Name"]} \n Возраст: {student["Age"]} лет \n Курс: {student["course"]}')
        print("Сложность: O(1) - доступ к элементам словаря и вывод")

    if num == '4':
        print('4.  Используя цикл, создать список, содержащий все целые числа от 1 до 10, и вывести его на экран.')
        spisok = []
        for i in range(10):
            spisok.append(i + 1)
        print(spisok)
        print("Сложность: O(N) - создание списка через цикл (где N - количество чисел в списке)")

    if num == '5':
        print('5.  Написать функцию, которая принимает список чисел и возвращает их сумму.')
        def sum_of_numbers(numbers):
            return sum(numbers)

        nums = list(map(int, input('Введите числа через пробел: ').split()))
        print("Сумма чисел в списке:", sum_of_numbers(nums))
        print("Сложность: O(N) - вычисление суммы элементов списка (где N - количество чисел в списке)")

    if num == '6':
        print('6.  Создать множество из 5 разных чисел, затем добавить в него новое число и вывести на экран.')

        m = {1, 2, 3, 4, 5}
        print(m)
        m.add(6)
        print(m)
        print("Сложность: O(1) - добавление элемента в множество")

    if num == '7':
        print('7.  Написать программу, которая принимает от пользователя целое число и выводит на экран его квадрат')
        number = int(input('Введите число:'))
        print(number * number)
        print("Сложность: O(1) - вычисление квадрата числа")

    if num == '8':
        print('8.  Создать словарь с пятью парами «ключ-значение», где ключи - названия фруктов, а значения - их цвета. Вывести значения всех ключей')
        fruits = {'apple': 'red', 'banana': 'yellow', 'pineapple': 'yellow', 'orange': 'orange', 'blueberry': 'blue'}
        for k in fruits:
            print(k, fruits[k])
        print("Сложность: O(N) - получение ключей из словаря (где N - количество пар ключ-значение в словаре)")

    if num == '9':
        def reverse_string(s):
            return s[::-1]

        print("Обратная строка:", reverse_string(input('Введите строку:')))
        # Сложность: O(N) - обратный порядок строки (где N - длина строки)

    if num == '10':
        print('10.  Создать список из 5 строк и заменить в нем третий элемент на новую строку')
        spisok = ['hello', 'world', 'meow', 'cat', 'dog']
        new = input('Введите новую строку:')
        spisok[2] = new
        print(spisok)
        print("Сложность: O(1) - замена элемента в списке")

    if num == '11':
        print("11. Создать кортеж из 6 элементов разных типов данных и вывести на экран тип каждого элемента.")
        mixed_tuple = (1, "текст", 3.14, True, [1, 2, 3], {"ключ": "значение"})
        for element in mixed_tuple:
            print(type(element))
        print("Сложность: O(N) - перебор элементов кортежа (где N - количество элементов в кортеже)")
    
    if num == '12':
        print("12. Написать программу, которая принимает от пользователя два числа и выводит их произведение.")
        num1 = int(input("Введите первое число: "))
        num2 = int(input("Введите второе число: "))
        print("Произведение чисел:", num1 * num2)
        print("Сложность: O(1) - вычисление произведения чисел")

    if num == '13':
        print("13. Создать словарь, содержащий информацию о книге (автор, название, год издания), и вывести эту информацию на экран.") 
        book = {"автор": "Толстой", "название": "Война и мир", "год издания": 1869}
        print("Информация о книге:", book)
        print("Сложность: O(1) - доступ к элементам словаря и вывод")

    if num == '14':
        print("14. Создать множество, содержащее названия 5 разных городов, затем удалить одно название и вывести оставшиеся.")
        cities_set = {"Москва", "Лондон", "Париж", "Нью-Йорк", "Токио"}
        print("Множество городов до удаления:", cities_set)
        cities_set.remove("Париж")
        print("Множество городов после удаления:", cities_set)
        print("Сложность: O(1) - удаление элемента из множества")

    if num == '15':
        print("15. Написать функцию, которая принимает список чисел и возвращает максимальное из них.")
        def max_in_list(numbers):
            return max(numbers)

        nums = list(map(int, input('Введите числа через пробел: ').split()))
        print("Максимальное число в списке:", max_in_list(nums))
        print("Сложность: O(N) - нахождение максимального элемента списка (где N - количество чисел в списке)")

    if num == '16':
        print("16. Создать список чисел от 1 до 20, а затем создать новый список, содержащий только четные числа из первого списка.")
        numbers = list(range(1, 21))
        even_numbers = [num for num in numbers if num % 2 == 0]
        print("Четные числа от 1 до 20:", even_numbers)
        print("Сложность: O(N) - создание списка и фильтрация четных чисел (где N - количество чисел в списке)")

    if num == '17':
        print("17. Написать программу, которая принимает от пользователя строку и выводит True"
              + ", если строка является палиндромом, и False в противном случае.")
        input_string = input("Введите строку: ")
        is_palindrome = input_string == input_string[::-1]
        print("Является ли строка палиндромом:", is_palindrome)
        print("Сложность: O(N) - проверка строки на палиндром (где N - длина строки)")
    
    if num == '18':
        print("18. Создать кортеж, содержащий 3 элемента различных типов данных, и распаковать его в отдельные переменные.")
        mixed_tuple = (10, "текст", 3.14)
        num, text, pi = mixed_tuple
        print("Распакованные значения:", num, text, pi)
        print("Сложность: O(1) - распаковка кортежа фиксированной длины")
    
    if num == '19':
        print("19. Написать функцию, которая принимает словарь с информацией о студенте (имя, возраст, курс) "
              + "и выводит эту информацию в форматированном виде.")
        def print_student(student):
            print(f"Имя: {student['имя']}\nВозраст: {student['возраст']}\nКурс: {student['курс']}")

        print_student({"имя": "Алексей", "возраст": 21, "курс": "Информатика"})
        print("Сложность: O(1) - вывод информации из словаря")

    if num == '20':
        print("20. Создать список и добавить в него 5 различных элементов. Вывести содержимое списка на экран.")
        my_list = []
        my_list.extend([1, "два", 3.0, "четыре", [5]])
        print("Содержимое списка:", my_list)
        print("Сложность: O(N) - добавление элементов в список (где N - количество добавляемых элементов)")

    num = input('Введите номер задачи (выход - exit):')
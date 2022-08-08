[TOC]

# Нейроинформатика

Данный репозиторий содержит лабораторные работы по курсу "Нейроинформатика". Все работы выполнены на языке программирования Python версии 3.9.11 с использованием библиотеки TensorFlow (Keras) версии 2.9.1. Со временем планируется добавление реализации с использованием библиотеки PyTorch.

Конфигурация машины: AMD Ryzen 5 5600H 3.30 ГГц 6 ядер / 12 потоков, 16 Гб ОЗУ. Для обучения моделей ГП не используется.

## ЛР1. Персептрон Розенблатта

В первой работе предлагается рассмотреть задачу классификации на примере персептрона Розенблатта представляющего собой простую математическую модель отдельного нейрона. Персептрон представляет собой сумматор с пороговой функцией Хевисайда. Модель персептрона в данной работе видоизменена так как TensorFlow не умеет работать с не дифференцируемыми функциями которой является пороговая функция Хевисайда. Пороговая функция заменена на функцию активации сигмоиду с предположением что любые значения выше 0,5 соответствуют значению 1, а любые значения меньше 0,5 соответствуют значению 0. Стоит также отметить что качество обучения модели сильно зависит от начальных условий - значений весовых коэффициентов и смещения.

Среднее время обучения модели: 15 с.

## ЛР3

В данной работе предлагается рассмотреть полносвязные нейронные сети прямого распространения в задаче классификации и аппроксимации. Работу условно можно разделить на две части: классификация и аппроксимация. В рамках задачи классификации предлагается обучить нейросетевую модель предсказывать принадлежность точки на двухмерной плоскости одной из трёх фигур. Стоит отметить что точка считается принадлежащей фигуре если она лежит на её границе, остальные области двухмерной плоскости не рассматриваются. Однако стоит отметить что рассмотрение каждой точки плоскости всё ещё представляет интерес так как позволяет заглянуть во внутреннее представление сети. Так как классов (фигур) на плоскости ровно три каждую точку пространства возможно легко преобразовать в пиксель цветового пространства RGB. Строгие красный, зелёный, синий цвета будут представлять области пространства принадлежащие одной из трёх фигур. Остальные цвета образуют переходные состояния.

Среднее время обучения модели: 30 с.



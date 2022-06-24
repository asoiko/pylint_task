# Pylint

Pylint — инструмент статического анализа исходного кода. Pylint проверяет исходный код на предмет ошибок, а также на предмет нарушения общепринятых правил оформления кода. 

## Установка 

Pylint не входит в стандартную поставку Python и должен быть установлен отдельно. Это можно сделать с помощью пакетного менеджера pip:

    $ pip install pylint

## Запуск

Основной способ запуска Pylint — из командной строки:

    $ pylint [options] modules_or_packages

Здесь:

* ```options``` — это необязательный список параметров запуска Pylint, 
* ```modules_or_packages``` — список модулей или пакетов, которые нужно проанализировать. 

Также можно проаназилировать отдельно взятый файл, например, filename.py:

    $ pylint filename.py

Здесь filename.py — имя файла, который будет проанализирован. 

## Конфигурация

Опции можно передать в Pylint двумя способами:
* через параметры командной строки;
* через конфигурационный файл.

Можно указать конфигурационный файл через параметр командной строки ```--rcfile``` Например:

    $ pylint --rcfile=pylintconfig filename.py

Если в командной строке не указан конфигурационный файл, то Pylint последовательно проверяет наличие конфигурации в следующих файлах:

1. ```pylintrc``` в текущей рабочей папке
2. ```.pylintrc``` в текущей рабочей папке
3. ```pyproject.toml``` в текущей рабочей папке, при условии, что этот файл содержит хотя бы одну секцию ```tool.pylint```. Файл ```pyproject.toml``` должен содержать секции, которые начинаются с ```tool.pylint.```, например: ```[tool.pylint.'MESSAGES CONTROL']```. 
4. ```setup.cfg``` в текущей рабочей папке, при условии, что этот файл содержит хотя бы одну секцию ```pylint```.
5. И другие места, полный перечень которых можно прочитать в [официальной документации](https://pylint.pycqa.org/en/latest/user_guide/usage/run.html#command-line-options). 

## Проверки и правила

Pylint группирует все проверки и правила в группы:

* (F)atal — ошибки из-за которых Pylint не может выполнить проверки до конца;
* (E)rror — вероятные ошибки;
* (W)arning — предупреждения;
* (C)onvention — оформление кода, форматирование, читаемость;
* (R)efactor — возможные рефакторинги, запахи кода;
* (I)nformation — информационные сообщения;

Все проверки имеют идентификатор, который начинается с буквы и содержит несколько цифр. Первая буква показывает к какой группе относится сообщение. Например, `E2502` — Error, `W1641` — Warning.

Рекомендуем ознакомиться с [полным перечнем проверок и их значений](https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html) в официальной документации. А также с [аннотированным списком проверок](https://vald-phoenix.github.io/pylint-errors/#list-of-errors). 

## Как читать сообщения

Рассмотрим файл `quick_sort.py`:

```python
def quickSort(arr):
    less = []
    pivotList = []
    more = []
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        for i in arr:
            if i < pivot:
                less.append(i)
            elif i > pivot:
                more.append(i)
            else:
                pivotList.append(i)
        less = quickSort(less)
        more = quickSort(more)
        return less + pivotList + more
```

Запустим Pylint с помощью команды в консоли:

    pylint quick_sort.py

Результат работы будет следующим:

```
************* Module quick_sort
quick_sort.py:22:0: C0304: Final newline missing (missing-final-newline)
quick_sort.py:1:0: C0114: Missing module docstring (missing-module-docstring)
quick_sort.py:1:0: C0116: Missing function or method docstring (missing-function-docstring)
quick_sort.py:1:0: C0103: Function name "quickSort" doesn't conform to snake_case naming style (invalid-name)
quick_sort.py:3:4: C0103: Variable name "pivotList" doesn't conform to snake_case naming style (invalid-name)
quick_sort.py:5:4: R1705: Unnecessary "else" after "return", remove the "else" and de-indent the code inside it (no-else-return)

------------------------------------------------------------------
Your code has been rated at 6.67/10 (previous run: 6.67/10, +0.00)
```

Здесь указан, к какому модулю относятся сообщения:

    ************* Module quick_sort

Далее следуют однострочные сообщения:

    quick_sort.py:22:0: C0304: Final newline missing (missing-final-newline)

Здесь:

* `quick_sort.py:22:0` — имя файла и номер строки (22) к которой относится сообщение;
* `C0304` — идентификатор сообщения;
* `Final newline missing (missing-final-newline)` — краткая аннотация сообщения. 

## Исправление ошибок

Программист последовательно анализирует отчет Pylint и справляет код. С рекомендациями по исправлению можно ознакомиться в [аннотированном справочнике](https://vald-phoenix.github.io/pylint-errors/#list-of-errors).

После исправления ошибок необходимо запустить еще раз Pylint, чтобы убедиться, что предупреждений больше нет. Также нужно запустить автоматические тесты, чтобы убедиться, что не были допущены случайные регрессии.

## Игонорирование предупреждений

Иногда полезно настроить Pylint таким образом, чтобы он игнорировал некоторые сообщения. Например, для этого кода:

```python
aaa = 24
```

Pylint сгенерирует предупреждение:

```
message.py:1:0: C0103: Constant name "aaa" doesn't conform to UPPER_CASE naming style (invalid-name)
```

И нам хотелось бы все-таки оставить именно `aaa` в качестве имени переменной. Для этого можно воспользоваться специальным мета-комментарием в конце строки, к которой относится сообщение:

```python
aaa = 24  # pylint: disable=invalid-name
```

# Практическое занятие

В результате успешного выполнения практического задания студент будет уметь:

* устанавливать Pylint;
* запускать Pylint для анализа исходного текста программы;
* анализировать отчет Pylint;
* исправлять некоторые ошибки;
* исключать некоторые проверки;

## Требования к программному обеспечению

* Операционная система: Windows, MacOS или Linux;
* Python 3.10 или старше
* Любой текстовый редактор


## Установите Pylint

Выполните в командной строке:

    pip install pylint

Сообщение об успешной установке должно содержать строку:

```
Successfully installed astroid-2.11.6 dill-0.3.5.1 isort-5.10.1 lazy-object-proxy-1.7.1 mccabe-0.7.0 platformdirs-2.5.2 pylint-2.14.2 tomli-2.0.1 tomlkit-0.11.0 wrapt-1.14.1
```

Можно выполнить проверку установки командой в консоли:

    pylint --version

Примерный ответ будет такой (версии интерпретатора и пакетов могут быть другими):

```
pylint 2.14.2
astroid 2.11.6
Python 3.10.1 (v3.10.1:2cd268a3a9, Dec  6 2021, 14:28:59) [Clang 13.0.0 (clang-1300.0.29.3)]
```

## Анализ файла

Проанализируйте сообщения Pylint для файла `triangle_overlap.py`:

```python
# Source: http://rosettacode.org/wiki/Determine_if_two_triangles_overlap#Python
from __future__ import print_function
import numpy as np


def CheckTriWinding(tri, allowReversed):
    trisq = np.ones((3, 3))
    trisq[:, 0:2] = np.array(tri)
    detTri = np.linalg.det(trisq)
    if detTri < 0.0:
        if allowReversed:
            a = trisq[2, :].copy()
            trisq[2, :] = trisq[1, :]
            trisq[1, :] = a
        else:
            raise ValueError("triangle has wrong winding direction")
    return trisq


def TriTri2D(t1, t2, eps=0.0, allowReversed=False, onBoundary=True):
    # Trangles must be expressed anti-clockwise
    t1s = CheckTriWinding(t1, allowReversed)
    t2s = CheckTriWinding(t2, allowReversed)

    if onBoundary:
        # Points on the boundary are considered as colliding
        chkEdge = lambda x: np.linalg.det(x) < eps
    else:
        # Points on the boundary are not considered as colliding
        chkEdge = lambda x: np.linalg.det(x) <= eps

    # For edge E of trangle 1,
    for i in range(3):
        edge = np.roll(t1s, i, axis=0)[:2, :]

        # Check all points of trangle 2 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (chkEdge(np.vstack((edge, t2s[0]))) and
                chkEdge(np.vstack((edge, t2s[1]))) and
                chkEdge(np.vstack((edge, t2s[2])))):
            return False

    # For edge E of trangle 2,
    for i in range(3):
        edge = np.roll(t2s, i, axis=0)[:2, :]

        # Check all points of trangle 1 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (chkEdge(np.vstack((edge, t1s[0]))) and
                chkEdge(np.vstack((edge, t1s[1]))) and
                chkEdge(np.vstack((edge, t1s[2])))):
            return False

    # The triangles collide
    return True


if __name__ == "__main__":
    t1 = [[0, 0], [5, 0], [0, 5]]
    t2 = [[0, 0], [5, 0], [0, 6]]
    print(TriTri2D(t1, t2), True)

    t1 = [[0, 0], [0, 5], [5, 0]]
    t2 = [[0, 0], [0, 6], [5, 0]]
    print(TriTri2D(t1, t2, allowReversed=True), True)

    t1 = [[0, 0], [5, 0], [0, 5]]
    t2 = [[-10, 0], [-5, 0], [-1, 6]]
    print(TriTri2D(t1, t2), False)

    t1 = [[0, 0], [5, 0], [2.5, 5]]
    t2 = [[0, 4], [2.5, -1], [5, 4]]
    print(TriTri2D(t1, t2), True)

    t1 = [[0, 0], [1, 1], [0, 2]]
    t2 = [[2, 1], [3, 0], [3, 2]]
    print(TriTri2D(t1, t2), False)

    t1 = [[0, 0], [1, 1], [0, 2]]
    t2 = [[2, 1], [3, -2], [3, 4]]
    print(TriTri2D(t1, t2), False)

    # Barely touching
    t1 = [[0, 0], [1, 0], [0, 1]]
    t2 = [[1, 0], [2, 0], [1, 1]]
    print(TriTri2D(t1, t2, onBoundary=True), True)

    # Barely touching
    t1 = [[0, 0], [1, 0], [0, 1]]
    t2 = [[1, 0], [2, 0], [1, 1]]
    print(TriTri2D(t1, t2, onBoundary=False), False)
```

Скопируйте отчет в файл `pylint_report.txt`.

## Исправление замечений Pylint

Проигнорируйте все сообщения, которые касаются переменной `a`. 

Исправьте остальные замечания. 

## Отчет

Для отчета по практическому заданию необходимо:

1. Создать публичный репозиторий на GitHub с названием pylint_task на основе репозитория https://github.com/1irs/pylint_practice Можно воспользоваться страницей Generate: https://github.com/1irs/pylint_practice/generate для создания своего репозитория по шаблону.
2. В отдельном коммите исправить ошибки и добавить файл `pylint_report.txt`
3. Прислать ссылку на репозиторий на obrizan@1irs.net

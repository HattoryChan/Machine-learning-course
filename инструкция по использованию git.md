# Инструкция по использованию git
Инструкция написана для абстрактного Иванова Ивана. Везде где написано Ivanov_Ivan нужно заменить на свои фамилию и имя.  
1. На странице репозитория нажимаем кнопку branch вводим в форму Ivanov_Ivan и нажимаем Create branch.  
![](https://github.com/Vlako/NapoleonMLCourse/blob/master/images/create_branch.png)  
2. Выполняем команду `git clone https://github.com/Vlako/NapoleonMLCourse -b Ivanov_Ivan`  
3. Переходим в папку с репозиторием `cd NapoleonMLCourse`
4. Прописываем в конфиг репозитория свои email и name  
`git config user.email "iivanov@fakemail.com"`  
`git config user.name "Ivan Ivanov"`  
5. Делаем домашнее задание
6. Фиксируем изменения файлов с домашним заданием `git add .`
7. Делаем коммит `git commit -m "Добавил решение домашнего задания 1"`
8. Отправлем свое решение на гитхаб `git push`
9. Перед тем как решать следующее задание получаем изменения с master ветки `git pull origin master`
10. Повторяем шаги начиная с шага 5

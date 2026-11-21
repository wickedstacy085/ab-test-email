# Данные

`hillstrom.csv` — открытый датасет Кевина Хиллстрома «MineThatData E-Mail Analytics and Data Mining
Challenge» (2008): 64 000 клиентов, которые покупали в последние 12 месяцев, случайно разделены
на три группы — мужская рассылка, женская рассылка, без рассылки (контроль). Отклик измерялся две недели.

Источник: http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
(опубликован автором для свободного анализа). Файл лежит в репозитории, чтобы сборка не зависела от сайта-источника.

| Поле | Описание |
|---|---|
| recency | месяцев с последней покупки |
| history_segment, history | сумма покупок за последний год (категория и $) |
| mens, womens | покупал мужские / женские товары (0/1) |
| zip_code | Urban / Surburban / Rural |
| newbie | новый клиент за 12 месяцев (0/1) |
| channel | Phone / Web / Multichannel |
| segment | группа эксперимента |
| visit, conversion, spend | визит на сайт, покупка, сумма покупки ($) за 2 недели |

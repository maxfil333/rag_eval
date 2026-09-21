## RU

Собрать RAG сегодня — вопрос вечера. Понять, стал ли он лучше после ваших правок — вопрос, на который у большинства команд нет ответа.

Знакомая картина: поменяли модель эмбеддингов, прогнали вручную десяток вопросов, ответы вроде стали складнее — закоммитили и пошли дальше. Через месяц кто-то спрашивает, зачем мы вообще меняли эмбеддинги, и ответить нечего.

Это не измерение. Это ощущение.

Я разбирался, как делать это нормально, и собрал разбор в двух частях. Здесь коротко расскажу, о какой задаче речь и что в этих частях будет.

Задача звучит просто.

У вас есть RAG. Вы меняете embeddings, vector store, reranker или стратегию чанкинга — и хотите объективно понимать, какая конфигурация лучше. Числом, а не на глаз.

Но как только берёшься за это всерьёз, задача распадается на две проблемы, и обе больше, чем выглядят.

Проблема 1. Нужен evaluation dataset.

Часть метрик считается прямо по логам работающего агента: вопрос, ответ, извлечённый контекст. Но самые полезные метрики требуют эталона — «идеального ответа». Взять его негде: разметка экспертами долгая и дорогая, а если просто попросить LLM сгенерировать вопросы по документам, получится набор гладких «What is X?», на которых ваш retrieval никогда не сломается. Реальный пользователь пишет коротко, с опечатками, не знает терминов — а иногда задаёт вопрос, ответ на который лежит сразу в двух документах.

Первая часть — про то, как собрать такой датасет на Ragas: граф знаний вместо списка чанков, роли пользователей, стили и длина запросов, вопросы с ответом в двух документах. С разбором решений, которые приходится принимать осознанно, и ловушек, которые не выпадают ни warning-ом, ни падением. Например, я довольно долго платил за эмбеддинги, которые в этом пайплайне не читает вообще никто.

Проблема 2. Нужны метрики.

Датасет сам по себе ничего не измеряет. Дальше надо понять, какие метрики отвечают за retrieval, а какие за генерацию; какие можно считать без эталона, а какие только с ним; и как по итоговым числам сравнивать конфигурации, чтобы вывод «стало лучше» был честным, а не шумом на шести примерах.

Это вторая часть — про Context Precision, Faithfulness, Answer Relevancy и Answer Correctness, и про то, как собрать из них сравнение конфигураций.

Полная версия — с кодом, замерами и разбором внутренностей библиотеки — в комментариях.

#RAG #LLM #MachineLearning #Ragas #LLMOps #DataEngineering

---

## EN

Standing up a RAG pipeline is an evening's work. Knowing whether your last change made it better is a question most teams cannot answer.

A familiar picture: you swap the embedding model, run a dozen questions by hand, the answers feel more coherent — you commit and move on. A month later someone asks why we changed the embeddings in the first place, and there is nothing to say.

That is not measurement. That is a feeling.

I went and figured out how to do it properly, and wrote it up in two parts. Here is the short version of what the task is and what those parts cover.

The task sounds simple.

You have a RAG system. You swap the embeddings, the vector store, the reranker or the chunking strategy — and you want to know objectively which configuration is better. As a number, not by eye.

But the moment you take it seriously, the task splits into two problems, and both are bigger than they look.

Problem 1. You need an evaluation dataset.

Some metrics can be computed straight from the logs of a running agent: the question, the answer, the retrieved context. But the most useful metrics need ground truth — an "ideal answer". And there is nowhere to get it: expert labelling is slow and expensive, while simply asking an LLM to generate questions from your documents produces a set of polished "What is X?" queries that your retrieval will never break on. Real users write short, write with typos, don't know the terminology — and sometimes ask a question whose answer lives in two documents at once.

Part one is about building exactly that dataset with Ragas: a knowledge graph instead of a flat list of chunks, user personas, query styles and lengths, questions that span two documents. Along with the decisions you have to make deliberately, and the traps that surface as neither a warning nor a crash. For quite a while, for instance, I was paying for embeddings that nothing in this pipeline reads at all.

Problem 2. You need metrics.

A dataset measures nothing on its own. Next you have to work out which metrics speak about retrieval and which about generation; which can be computed without ground truth and which cannot; and how to compare configurations from the resulting numbers so that "it got better" is an honest conclusion rather than noise across six examples.

That is part two — Context Precision, Faithfulness, Answer Relevancy, Answer Correctness, and how to turn them into a comparison of configurations.

Full version, with code, measurements and a walkthrough of the library internals, is in the comments.

#RAG #LLM #MachineLearning #Ragas #LLMOps #DataEngineering

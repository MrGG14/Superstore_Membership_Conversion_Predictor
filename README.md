# Superstore Membership Conversion Predictor

Context:

A superstore is planning for the year-end sale. They want to launch a new offer - gold membership, that gives a 20% discount on all purchases, for only 499
 on other days. It will be valid only for existing customers and the campaign through phone calls is currently being planned for them. The management feels that the best way to reduce the cost of the campaign is to make a predictive model which will classify customers who might purchase the offer.

Objective:

The superstore wants to predict the likelihood of the customer giving a positive response and wants to identify the different factors which affect the customer's response. You need to analyze the data provided to identify these factors and then build a prediction model to predict the probability of a customer will give a positive response.

This data was gathered during last year's campaign. Data description is as follows;

* Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise

* ID - Unique ID of each customer

* Year_Birth - Age of the customer

* Complain - 1 if the customer complained in the last 2 years

* Dt_Customer - date of customer's enrollment with the company

* Education - customer's level of education

* Marital - customer's marital status

* Kidhome - number of small children in customer's household

* Teenhome - number of teenagers in customer's household

* Income - customer's yearly household income

* MntFishProducts - the amount spent on fish products in the last 2 years

* MntMeatProducts - the amount spent on meat products in the last 2 years

* MntFruits - the amount spent on fruits products in the last 2 years

* MntSweetProducts - amount spent on sweet products in the last 2 years

* MntWines - the amount spent on wine products in the last 2 years

* MntGoldProds - the amount spent on gold products in the last 2 years

* NumDealsPurchases - number of purchases made with discount

* NumCatalogPurchases - number of purchases made using catalog (buying goods to be shipped through the mail)

* NumStorePurchases - number of purchases made directly in stores

* NumWebPurchases - number of purchases made through the company's website

* NumWebVisitsMonth - number of visits to company's website in the last month

* Recency - number of days since the last purchase

La variable a predecir es "Response", "1" indica que el cliente acepto la oferta y "0" que no la aceptó. Por tanto nos encontramos ante un problema de clasificacion binario, que resolveremos mediante un modelo de aprendizaje supervisado usando 4 modelos distintos:

* KNN

* DecisionTree

* Logistic Regression

* SVM

## PREPROCESSING
Vemos que tenemos 2240 casos y 22 columnas. Tan solo hay una columna con nulos (Income).

La mayoria de las variables son numericas, salvo: "Education", "Marital_status" y "Dt_Customer". Tendremos que estudiar que hacer con estas variables para que tomen valores numericos.
Tan solo el 14,91% de los casos corresponden a la clase positiva, esto hara que en el modelo haya overfitting de los valores negativos.

Trataremos de solventarlo a la hora de establecer la metrica al hacer CV para que maximice el valor de f1-score.
Hay 24 valores ausentes en la variable 'Income'. Observamos las entradas que tienen estos valores ausentes y vemos que el resto de valores parecen ser reales y con sentido. A pesar de ser una proporcion pequeña dentro de nuestro numero de observaciones, vamos a tratar de imputar los valores en vez de suprimir estas filas.
Al observar los posibles valores de la variable "Marital_Status" vemos que hay valores incorrectos ('YOLO', 'Alone', 'Absurd'). A YOLO y Alone los establecemos como 'single', y a 'Absurd' como nulo. Posteriormente haremos One-Hot-Encodign y el 'Absurd' se codificara como 0´s. Una vez ya hemos corregido los posibles valores de las variables categoricas procedemos a crear Dummies.
Procedemos a modificar ciertas variables para que tengan más sentido de cara al modelo:

* En vez de tener una varibale con la fecha en la que el cliente se dio de alta tendremos una con la antiguedad en años del cliente.

* En vez de tener la fecha de nacimiento del cliente tendremos una variable que corresponda a la edad.
Eliminamos las columnas no necesarias como Id, Dt_Customer y Year_Birth

Con "describe" vemos informacion sobre los valores que toman las variables. Vemos que todas parecen tener valores reales salvo en la variable "Age", en la que vemos que la edad maxima son 130 años, imposible. Por tanto vemos las filas que tienen una edad considerable (mayores que 100 por ejemplo) y vemos que hacer con dichos valores.

Vemos que las edades son de 123, 126 y 130 años, valores surrealistas. Sin embargo el resto de valores de dichas filas tiene sentido, por tanto haremos uso de KNN Imputer.

Creamos 3 variables más que considero que podrian ser relevantes:

* total_spent: valor total gastado (suma de vinos, frutas, carne, pescado, dulces y oro).

* deals_ratio: proporcion de compras en las que se ha usado algun tipo de descuento.

* spent_ratio: proporcion de dinero gastado en tienda en comparacion con los ingresos del cliente.



## COMPARACION MODELOS
En la siguiente tabla podemos ver las metricas asociadas a la clase "1". El resto de metricas correspondientes a la clase "0" no nos interesan pues seran valores altos ya que estan muy desbalanceadas las clases. Como el modelo en la mayoria de casos apostara por que la clase correcta sea "0", y la mayoria de casos pertenecen a la clase "0", las metricas seran de esta clase seran altas, dando una falsa sensacion de acierto del modelo.

Observamos ejecutando todos los algoritmos en varias ocasiones que los modelos en general no dan resultados muy distintos, aun asi podemos ver que hay modelos ligeramente mejores que otros.

Como podemos ver el "recall" es muy bajo siempre debido al desbalanceo entre clases, tan solo identificamos como positivos aproximadamente el 30% del total. Además la precision es de un 50% de media, significando que de todos los casos que identificamos como positivos, tan solo la mitad lo son realmente. Se puede concluir que el modelo esta sesgado a etiquetar como negativo.

En cuanto al "mejor modelo" es dificil elegir uno a ciencia cierta. Diremos que un modelo es mejor que otro si su valor f1 es mayor que el del otro. Esto lo hacemos asi pues f1 es una media armonica de "precission" y "recall", penalizando ambos por igual. Por tanto, podemos decir que los mejores modelos serian:

* Logistic Regression

* KNN (la version Grid parece dar mejores resultados que la version Random por algun motivo)

* Decision Tree


|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| KNN_Grid     | 0.463415  | 0.283582 | 0.351852 | 67      |
| KNN_Rand     | 0.470588  | 0.358209 | 0.406780 | 67      |
| Dec_Tree_Grid| 0.369565  | 0.253731 | 0.300885 | 67      |
| Dec_Tree_Rand| 0.369565  | 0.253731 | 0.300885 | 67      |
| LogReg_Grid  | 0.536585  | 0.328358 | 0.407407 | 67      |
| LogReg_Rand  | 0.536585  | 0.328358 | 0.407407 | 67      |
| SVM_Grid     | 0.555556  | 0.223881 | 0.319149 | 67      |
| SVM_Rand     | 0.555556  | 0.223881 | 0.319149 | 67      |




En base a los resultados anteriores y a distintas ejecuciones de cada modelo, y en vista a que solo podemos elegir un modelo, elegiremos como mejor modelo el generado por Logistic Regression (grid o random, los resultados suelen ser iguales) al tener el mayor valor de "f1", a pesar de no ser muy alto tampoco.

Visualizamos cual es el mejor modelo hallado tras hacer cross-validation probando con distintos parametros (lo dejamos automatizado, puede ser que haya un caso concreto en el que Logistic Regression no sea el mejor modelo, pero generalmente lo será)
El mejor modelo es: LogisticRegression(solver='sag')

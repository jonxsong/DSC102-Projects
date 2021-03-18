"""
Jon Zhang
Brent Min
"""
import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics

# %load -s task_1 assignment2.py
def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    data = review_data.groupBy(F.col(asin_column)).agg(F.avg(F.col(overall_column)).alias(mean_rating_column), 
                                                       F.count("*").alias(count_rating_column))
    
    merged = product_data.join(data, on=asin_column, how= 'left')
    
    aggregate_func = merged.agg(F.count("*"), 
                                F.avg(F.col(mean_rating_column)), 
                                F.variance(F.col(mean_rating_column)), 
                                F.sum(F.isnull(F.col(mean_rating_column)).astype("int")), 
                                F.avg(F.col(count_rating_column)), 
                                F.variance(F.col(count_rating_column)), 
                                F.sum(F.isnull(F.col(count_rating_column)).astype("int"))).collect()[0]
    
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:

    res['count_total'] = aggregate_func[0]
    res['mean_meanRating'] = aggregate_func[1]
    res['variance_meanRating'] = aggregate_func[2]
    res['numNulls_meanRating'] = aggregate_func[3]
    res['mean_countRating'] = aggregate_func[4]
    res['variance_countRating'] = aggregate_func[5]
    res['numNulls_countRating'] = aggregate_func[6]
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    ### category
    data = product_data.withColumn(category_column, 
                                   F.when(F.col(categories_column)[0][0] == '', None).otherwise(F.col(categories_column)[0][0]))
    category_nulls = data.filter(F.col(category_column).isNull()).count()
    category_distinct = data.agg(F.countDistinct(F.col(category_column))).head()[0]
    
    ### salesRank and salesCategory
    key_and_values = data.select(asin_column,
                                 category_column,
                                 F.map_keys(salesRank_column)[0].alias(bestSalesCategory_column),
                                 F.map_values(salesRank_column)[0].alias(bestSalesRank_column))
    
    mean_of_salesRank = key_and_values.select(F.avg(F.col(bestSalesRank_column))).head()[0]
    variance_of_salesRank = key_and_values.select(F.variance(F.col(bestSalesRank_column))).head()[0]
    
    salesCategory_nulls = key_and_values.filter(F.col(bestSalesCategory_column).isNull()).count()
    salesCategory_distinct = key_and_values.agg(F.countDistinct(F.col(bestSalesCategory_column))).head()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:

    res['count_total'] = data.count()
    res['mean_bestSalesRank'] = mean_of_salesRank
    res['variance_bestSalesRank'] = variance_of_salesRank
    res['numNulls_category'] = category_nulls
    res['countDistinct_category'] = category_distinct
    res['numNulls_bestSalesCategory'] = salesCategory_nulls
    res['countDistinct_bestSalesCategory'] = salesCategory_distinct

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


# %load -s task_3 assignment2.py
def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    data = product_data.select(F.col(asin_column), F.explode(F.col(related_column))).filter(F.col('key')==attribute)

    data = data.select(F.col(asin_column), F.explode_outer(F.col('value')))

    first_join = product_data.select(F.col(asin_column).alias("col"), F.col(price_column).alias("prices"))
    
    merged = data.join(first_join, on="col", how='left')
    
    merged = merged.groupby(F.col(asin_column)).agg(F.avg('prices').alias(meanPriceAlsoViewed_column), F.count('*').alias(countAlsoViewed_column))

    merged = merged.withColumn(countAlsoViewed_column, F.when(F.col(countAlsoViewed_column)==0, None).otherwise(F.col(countAlsoViewed_column)))
    
    count_total = product_data.count()
    
    numNulls_meanPriceAlsoViewed = count_total - merged.where((merged[meanPriceAlsoViewed_column].isNotNull())).count()
    
    numNulls_countAlsoViewed = count_total - merged.where((merged[countAlsoViewed_column].isNotNull())).count()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res['count_total']= count_total
    res['mean_meanPriceAlsoViewed'] = merged.select(F.avg(F.col(meanPriceAlsoViewed_column))).head()[0]
    res['variance_meanPriceAlsoViewed'] = merged.select(F.variance(F.col(meanPriceAlsoViewed_column))).head()[0]
    res['numNulls_meanPriceAlsoViewed'] = numNulls_meanPriceAlsoViewed
    res['mean_countAlsoViewed'] = merged.select(F.avg(F.col(countAlsoViewed_column))).head()[0]
    res['variance_countAlsoViewed'] = merged.select(F.variance(F.col(countAlsoViewed_column))).head()[0]
    res['numNulls_countAlsoViewed'] = numNulls_countAlsoViewed

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    casted_data = product_data.withColumn(price_column, F.col(price_column).cast(T.FloatType()))
    
    mean_imputer = M.feature.Imputer(strategy='mean',inputCols=[price_column], outputCols=[meanImputedPrice_column])
    mean_model = mean_imputer.fit(casted_data)
    meanImputedPrice = mean_model.transform(casted_data.select('asin', price_column))
    
    mean_meanImputedPrice = meanImputedPrice.select(F.avg(F.col(meanImputedPrice_column))).head()[0]
    variance_meanImputedPrice = meanImputedPrice.select(F.variance(F.col(meanImputedPrice_column))).head()[0]
    numNulls_meanImputedPrice = meanImputedPrice.filter(F.col(meanImputedPrice_column).isNull()).count()
    
    median_imputer = M.feature.Imputer(strategy='median',inputCols=[price_column], outputCols=[medianImputedPrice_column])
    median_model = median_imputer.fit(casted_data)
    medianImputedPrice = median_model.transform(meanImputedPrice)
    
    mean_medianImputedPrice = medianImputedPrice.select(F.avg(F.col(medianImputedPrice_column))).head()[0]
    variance_medianImputedPrice = medianImputedPrice.select(F.variance(F.col(medianImputedPrice_column))).head()[0]
    numNulls_medianImputedPrice = medianImputedPrice.filter(F.col(medianImputedPrice_column).isNull()).count()
    
    unknownImputedTitle = product_data.select(F.col(title_column).alias(unknownImputedTitle_column)).fillna('unknown').replace('','unknown')
    numUnknowns_unknownImputedTitle = unknownImputedTitle.filter(F.col(unknownImputedTitle_column)=='unknown').count()


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    
    res['count_total'] = medianImputedPrice.count()
    res['mean_meanImputedPrice'] = mean_meanImputedPrice
    res['variance_meanImputedPrice'] = variance_meanImputedPrice
    res['numNulls_meanImputedPrice'] = numNulls_meanImputedPrice
    res['mean_medianImputedPrice'] = mean_medianImputedPrice
    res['variance_medianImputedPrice'] = variance_medianImputedPrice
    res['numNulls_medianImputedPrice'] = numNulls_medianImputedPrice
    res['numUnknowns_unknownImputedTitle'] = numUnknowns_unknownImputedTitle

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    step = product_processed_data.select(title_column, F.lower(F.col(title_column)).alias('temp'))
    
    split = step.withColumn(titleArray_column, F.split(F.col('temp'), ' ').cast("array<string>"))
    
    word2vec = M.feature.Word2Vec(minCount=100, vectorSize=16, seed=SEED, numPartitions=4, inputCol=titleArray_column, outputCol=titleVector_column)
    model = word2vec.fit(split)

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:

    res['count_total'] = split.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    count_total = product_processed_data.count()

    indexer = M.feature.StringIndexer(inputCol=category_column, 
                                             outputCol=categoryIndex_column, 
                                             handleInvalid="error",
                                             stringOrderType="frequencyDesc").fit(product_processed_data).transform(product_processed_data)
    
    ohe = M.feature.OneHotEncoderEstimator(inputCols=[categoryIndex_column], 
                                                      outputCols=[categoryOneHot_column], 
                                                      dropLast=False).fit(indexer).transform(indexer)
 
    pca = M.feature.PCA(k=15, inputCol=categoryOneHot_column, 
                        outputCol=categoryPCA_column).fit(ohe).transform(ohe)
    
    
    meanVector_categoryOneHot = ohe.agg(M.stat.Summarizer.mean(F.col(categoryOneHot_column)).alias('ohe'))
    
    meanVector_categoryPCA = pca.agg(M.stat.Summarizer.mean(F.col(categoryPCA_column)).alias('pca'))

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res['count_total'] = count_total
    res['meanVector_categoryOneHot'] = meanVector_categoryOneHot.head().asDict()["ohe"]
    res['meanVector_categoryPCA'] = meanVector_categoryPCA.head().asDict()["pca"]

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    tree_regression = M.regression.DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol='overall') 
    model = tree_regression.fit(train_data)
    
    predicted = model.transform(test_data)
    
    # -------------------------------------------------------------------------
    evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction', labelCol='overall', metricName='rmse')
    RMSE = evaluator.evaluate(predicted)
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    
    res['test_rmse'] = RMSE

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    tree_regression = M.regression.DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol='overall') 
    
    evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction', labelCol='overall', metricName='rmse')
    
    tuning_grid = M.tuning.ParamGridBuilder().addGrid(tree_regression.maxDepth, [5,7,9,12]).build()
    
    tvs = M.tuning.TrainValidationSplit(estimator = tree_regression, estimatorParamMaps = tuning_grid, evaluator = evaluator, trainRatio=0.75)

    model = tvs.fit(train_data)
    
    best_predictions = model.bestModel.transform(test_data)
    
    test_RMSE = evaluator.evaluate(best_predictions)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    
    res['test_rmse'] = test_RMSE
    
    for name, rmse in zip(
        ['valid_rmse_depth_5', 'valid_rmse_depth_7', 'valid_rmse_depth_9', 'valid_rmse_depth_12'],
        model.validationMetrics
    ):
        res[name] = rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------


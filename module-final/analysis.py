import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from collections import Counter 
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

RECIPES = pd.concat([
    pd.read_csv("RAW_recipes.csv", usecols=["id", "tags"], index_col="id"), 
    pd.read_csv("PP_recipes.csv", usecols=["id", "ingredient_ids"], index_col="id")], 
    axis= 1).dropna()

def jaccard_similarity(recipe1, recipe2): 
    intersection = recipe1.intersection(recipe2)
    union = recipe1.union(recipe2)
    return len(intersection)/len(union)

def predict_tags(target_recipe_id, k): 
    for idx, recipe in RECIPES.iterrows():
        if idx == target_recipe_id:
            target_recipe = recipe
            break
    target_ingredients = {ingredient for ingredient in eval(target_recipe["ingredient_ids"])}

    similarities = []
    for idx, recipe in RECIPES.iterrows():
        if idx != target_recipe_id:
            recipe_ingredients = {ingredient for ingredient in eval(recipe["ingredient_ids"])}
            similarity = jaccard_similarity(target_ingredients, recipe_ingredients)
            similarities.append((similarity, eval(recipe["tags"])))

    similarities.sort(reverse=True)

    top_k_tags = []
    for similarity, tags in similarities[:k]:
        for tag in tags:
            top_k_tags.append(tag)

    tag_counts = Counter(top_k_tags)
    max_count = max(tag_counts.values())
    # print(top_k_tags)
    # print(tag_counts)
    # print(similarities[:k])
    most_common_tags = []

    for tag, count in tag_counts.items():
        if count == max_count:
            most_common_tags.append(tag)
            
    print( "\nRecipe ID: ", target_recipe_id)
    print("\nPredicted Tags: ", most_common_tags)
    print("\nGround Truth: ", target_recipe["tags"])
    print("\nCorrect Tags: ", list(set(eval(target_recipe["tags"])) & set(most_common_tags)))
    print("\nIncorrect Tags: ", list(set(most_common_tags) - set(eval(target_recipe["tags"]))), "\n")

def get_clusters():
    NUTRITION = []
    RAW_RECIPES = pd.read_csv("RAW_recipes.csv", usecols =["id", "name", "nutrition"], index_col="id" )
    rec_names = RAW_RECIPES["name"].to_dict()

    for val in RAW_RECIPES["nutrition"]:
        NUTRITION.append(json.loads(val))

    columns = ["Calories", "Total Fat", "Sugar", "Sodium", "Protein", "Sat Fat", "Carbs"]
    DF = pd.DataFrame(NUTRITION, columns= columns, index=RAW_RECIPES.index)
    # DF = pd.DataFrame(NUTRITION, columns= columns, index=RAW_RECIPES.index).sample(n=2000)
    norm_DF = pd.DataFrame(normalize(DF, norm= "max", axis=0), columns=columns, index=DF.index)
    pca = PCA(n_components = 2)
    pca.fit(norm_DF)
    pca_DF = pd.DataFrame(pca.transform(norm_DF), columns=["x", "y"], index=norm_DF.index)

    # Select DF or normalized DF.
    cur_DF = pca_DF 

    # Elbow Method
    inertia = []
    K = range(1, 15)
    for x in K:
        model = KMeans(n_clusters = x).fit(cur_DF)
        model.fit(cur_DF)
        inertia.append(model.inertia_)

    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()

    kmeans = KMeans(n_clusters=4, max_iter=10000)

    # Assign actors to clusters and append to dataframe.
    cluster_labels= kmeans.fit_predict(cur_DF)
    cur_DF["cluster"] = cluster_labels
    cur_DF["name"] = RAW_RECIPES["name"]

    for col in columns:
        cur_DF[col] = DF[col]
        
    # Print number of actors per cluster
    print("Number of recipes per cluster")
    print(cur_DF["cluster"].value_counts())

    # Group actors by cluster number
    grouped = cur_DF.groupby("cluster") 

    # Print 5 random samples from each cluster
    for cluster, group in grouped: 
        print(f"\nCluster {cluster}:")
        sample_recipes = group.sample(n=5).index
        for recipe in sample_recipes:
            print(rec_names[recipe])

    plt.figure(figsize=(40,24))
    plt.xscale("log")
    plt.yscale("log")
    plt.markeredgecolor = "none"
    plt.scatter(cur_DF["x"], cur_DF["y"], s = [10] * len(cur_DF["x"]), c = cur_DF["cluster"])
    plt.show()

predict_tags(25274, 4)
predict_tags(190891, 4)
predict_tags(160027, 4)
predict_tags(322059, 4)
predict_tags(271216, 4)

get_clusters()

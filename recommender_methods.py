def collaborative_filtering_user_based(rm, df_user_item, user_id, k, n):
    """
    performes user-based collaborative_filtering on a category and key word rating matrix. It returns the k-closest user
    to a given user read articles that the given user has not read.
    """
    # Compute peer group of the user

    user_id_row = rm.loc[user_id]
    with_pearson = rm.corrwith(user_id_row, axis=1)

    sorted_pearson = with_pearson.sort_values(ascending=False)
    k_closest = sorted_pearson[1: k + 1]


    # Find articles user_id has not read
    user_articles = df_user_item.loc[user_id]

    user_unread_articles = user_articles[user_articles != 1]

    # Find articles k users have read
    k_user_read_articles = df_user_item.loc[k_closest.index[0]]
    k_user_read_articles = k_user_read_articles[k_user_read_articles != 0]

    for i in range(1, len(k_closest)):
        temp = df_user_item.loc[k_closest.index[i]]
        temp = temp[temp != 0]
        k_user_read_articles = k_user_read_articles.combine(temp, max, fill_value=0)

    # Find intersection between given user's not read articles and k users read articles
    idx1 = user_unread_articles.index
    idx2 = k_user_read_articles.index

    recommended_articles = idx2.intersection(idx1)

    # Only recomend n-articles
    recommended_articles = recommended_articles[:n]

    # returns list with articles the given user will like
    return {"user_id": user_id, "articles": recommended_articles}





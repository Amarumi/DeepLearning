# Load existing word2vec model
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('nkjp.txt', binary=False)

# Define embedding matrix
embedding_matrix = word2vec_model.wv.syn0

# Choose 10 words from nkjp.txt dictonary, index = 320 346 456 455 1104 1105 1330 897 981 1206
word2index = {token: token_index for token_index, token in enumerate(word2vec_model.index2word)}
tsne_list =[word2index['pieniądze'],
           word2index['człowiek'],
           word2index['św'],
           word2index['dyrektor'],
           word2index['Paweł'],
           word2index['Ewa'],
           word2index['śmierci'],
           word2index['rząd'],
           word2index['kościoła'],
           word2index['Jezus']]

# Create list of choosen words
words = []

for i in range(len(tsne_list)):
    words.append(word2vec_model.index2word[tsne_list[i]])
print(words)

# Reduct dimension and plot 2d area
tsne_2d = TSNE(n_components=2).fit_transform(embedding_matrix[tsne_list,:])

for i,type in enumerate(words):
    x = tsne_2d[i, 0]
    y = tsne_2d[i, 1]
    plt.scatter(x, y, marker='o')
    plt.text(x+1.5, y+1.5, type, fontsize=9)
plt.show()

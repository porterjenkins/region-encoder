
def write_embeddings(arr, n_nodes, fname):

    h_dim = arr.shape[1]

    with open(fname, 'w') as f:
        f.write("{} {} \n".format(n_nodes, h_dim))

        # for cntr, embedding_vector in enumerate(arr):
        for region_idx in range(n_nodes):
            embedding_vector = arr[region_idx, :]
            f.write("{} ".format(region_idx))

            cntr = 0
            for element in embedding_vector:
                if cntr == (h_dim - 1):
                    f.write("{}".format(element))
                else:
                    f.write("{} ".format(element))

                cntr += 1

            f.write("\n")
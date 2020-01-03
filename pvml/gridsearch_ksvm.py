import ksvm


def grid_search_ksvm_rbf(Xtrain, Ytrain, Xval, Yval, kfun, lambdas,
                         kparams, lr=1e-3, steps=1000):
    best_accuracy = 0
    best_alpha = None
    best_b = None
    best_kparam = None
    results = []
    for lambda_ in lambdas:
        results.append([])
        for kparam in kparams:
            alpha, b, loss = ksvm.ksvm_train(Xtrain, Ytrain, kfun,
                                             kparam, lambda_, lr=lr,
                                             steps=steps)
            T = ksvm.ksvm_inference(Xval, Xtrain, alpha, b, kfun,
                                    kparam)
            acc = accuracy(Yval, T)
            results[-1].append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_lambda_ = lambda_
                best_kparam = kparam
                best_alpha = alpha
                best_b = b
    return (best_accuracy, best_lambda_, best_kparam, best_alpha,
            best_b, results)


def accuracy(Y1, Y2):
    return (Y1 == Y2).mean()


def _demo():
    import sys
    import demo
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1]
    gammas = [0.1, 0.3, 1, 3, 10, 30]
    Xtrain, Ytrain = demo.load_data(sys.argv[1], True)
    Xval, Yval = demo.load_data(sys.argv[2], True)
    res = grid_search_ksvm_rbf(Xtrain, Ytrain, Xval, Yval, "rbf",
                               lambdas, gammas, steps=10000)
    acc, lambda_, gamma, alpha, b, results = res
    for row in results:
        for col in row:
            print("%.4f" % col, end=" ")
        print()
    print("Best validation accuracy:", acc)
    print(alpha)
    print(b)


if __name__ == "__main__":
    _demo()

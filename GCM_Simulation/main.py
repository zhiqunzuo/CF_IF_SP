import numpy as np
import torch
import torch.optim as optim

from data_generation import sample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, pairwise
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main():
    # generate simulated data
    N = 10000
    d_x = 5
    d_u = 5
    
    a, x, y, w_u, cov_x, w_a, b, c_a, c_x = sample(N, d_u, d_x)
    
    # compute mu
    C_1 = np.linalg.inv(np.dot(np.dot(w_u.T, np.linalg.inv(cov_x)), w_u) + np.eye(d_u))
    C_2 = np.dot(np.dot(w_u.T, np.linalg.inv(cov_x)), (x.T - np.dot(w_a, a.reshape(1, -1)) - b[:, np.newaxis]))
    mu = (np.dot(C_1, C_2)).T
    
    cf_mses, uf_mses, if_mses = [], [], []
    cf_accs, uf_accs, if_accs = [], [], []
    cf_cofs, uf_cofs, if_cofs = [], [], []
    cf_tes, uf_tes, if_tes = [], [], []
    for seed in (42, 47):
        X = np.concatenate([a.reshape(-1, 1), x, mu], axis=1)
        cX = np.concatenate([c_a.reshape(-1, 1), c_x, mu], axis=1)
        X_train, X_test, cX_train, cX_test, y_train, y_test = train_test_split(X, cX, y, 
                                                                               test_size=0.2, random_state=seed)
        
        # CF method
        model = LinearRegression()
        model.fit(X_train[:, -d_u:], y_train)
        y_pred_train = model.predict(X_train[:, -d_u:])
        y_pred = model.predict(X_test[:, -d_u:])
        mse = mean_squared_error(y_test, y_pred)
        cf_mses.append(mse)
        
        c_y_pred = model.predict(cX_test[:, -d_u:])
        te = np.mean(np.abs(y_pred - c_y_pred))
        cf_tes.append(te)
        
        classifier = SVC()
        classifier.fit(y_pred_train.reshape(-1, 1), X_train[:, 0])
        a_pred = classifier.predict(y_pred.reshape(-1, 1))
        acc = accuracy_score(X_test[:, 0], a_pred)
        cf_accs.append(acc)
        
        x_test_distance = pairwise.euclidean_distances(X_test[:, :-d_u], X_test[:, :-d_u])
        y_test_distance = pairwise.euclidean_distances(y_pred.reshape(-1, 1), y_pred.reshape(-1, 1))
        if_cof = np.mean(y_test_distance / (x_test_distance + 1e-7))
        variance = np.std(y_test_distance / (x_test_distance + 1e-7))
        cf_cofs.append(if_cof)
        
        # UF method
        model = LinearRegression()
        model.fit(X_train[:, :-d_u], y_train)
        y_pred_train = model.predict(X_train[:, :-d_u])
        y_pred = model.predict(X_test[:, :-d_u])
        mse = mean_squared_error(y_test, y_pred)
        uf_mses.append(mse)
        
        c_y_pred = model.predict(cX_test[:, :-d_u])
        te = np.mean(np.abs(y_pred - c_y_pred))
        uf_tes.append(te)
        
        classifier = SVC()
        classifier.fit(y_pred_train.reshape(-1, 1), X_train[:, 0])
        a_pred = classifier.predict(y_pred.reshape(-1, 1))
        acc = accuracy_score(X_test[:, 0], a_pred)
        uf_accs.append(acc)
        
        x_test_distance = pairwise.euclidean_distances(X_test[:, :-d_u], X_test[:, :-d_u])
        y_test_distance = pairwise.euclidean_distances(y_pred.reshape(-1, 1), y_pred.reshape(-1, 1))
        if_cof = np.mean(y_test_distance / (x_test_distance + 1e-7))
        variance = np.std(y_test_distance / (x_test_distance + 1e-7))
        uf_cofs.append(if_cof)
        
        #IF method
        model = LinearRegression()
        model.fit(X_train[:, :-d_u], y_train)
        y_pred_train = model.predict(X_train[:, :-d_u])
        y_pred = model.predict(X_test[:, :-d_u])
        
        c_y_pred = model.predict(cX_test[:, :-d_u])
        
        lam = 1
        
        W_train = np.exp(-pairwise.euclidean_distances(X_train[:, :-d_u], X_train[:, :-d_u]))
        L_train = np.diag(np.sum(W_train, axis=-1)) - W_train
        y_fair_train = np.dot(np.linalg.inv(np.eye(X_train.shape[0]) + lam * (L_train + L_train.T) / 2), y_pred_train)
        
        W = np.exp(-pairwise.euclidean_distances(X_test[:, :-d_u], X_test[:, :-d_u]))
        L = np.diag(np.sum(W, axis=1)) - W
        y_fair = np.dot(np.linalg.inv(np.eye(X_test.shape[0]) + lam * (L + L.T) / 2), y_pred)
        
        mse = mean_squared_error(y_test, y_fair)
        if_mses.append(mse)
        
        c_W = np.exp(-pairwise.euclidean_distances(cX_test[:, :-d_u], cX_test[:, :-d_u]))
        c_L = np.diag(np.sum(c_W, axis=1)) - c_W
        c_y_fair = np.dot(np.linalg.inv(np.eye(X_test.shape[0]) + lam * (c_L + c_L.T) / 2), c_y_pred)
        
        te = np.mean(np.abs(y_fair - c_y_fair))
        if_tes.append(te)
        
        classifier = SVC()
        classifier.fit(y_fair_train.reshape(-1, 1), X_train[:, 0])
        a_pred = classifier.predict(y_fair.reshape(-1, 1))
        acc = accuracy_score(X_test[:, 0], a_pred)
        if_accs.append(acc)
        
        y_test_distance = pairwise.euclidean_distances(y_fair.reshape(-1, 1), y_fair.reshape(-1, 1))
        x_test_distance = pairwise.euclidean_distances(X_test[:, :-d_u], X_test[: , :-d_u])
        if_cof = np.mean(y_test_distance / (x_test_distance + 1e-7))
        variance = np.std(y_test_distance / (x_test_distance + 1e-7))
        if_cofs.append(if_cof)
    
    print("CF")
    print("mse = {} + {}, acc = {} + {}, if = {} + {}, te = {} + {}".format(
        np.mean(cf_mses), np.std(cf_mses),
        np.mean(cf_accs), np.std(cf_accs),
        np.mean(cf_cofs), np.std(cf_cofs),
        np.mean(cf_tes), np.std(cf_tes)))
    print("UF")
    print("mse = {} + {}, acc = {} + {}, if = {} + {}, te = {} + {}".format(
        np.mean(uf_mses), np.std(uf_mses),
        np.mean(uf_accs), np.std(uf_accs),
        np.mean(uf_cofs), np.std(uf_cofs),
        np.mean(uf_tes), np.std(uf_tes)))
    print("GLIF")
    print("mse = {} + {}, acc = {} + {}, if = {} + {}, te = {} + {}".format(
        np.mean(if_mses), np.std(if_mses),
        np.mean(if_accs), np.std(if_accs),
        np.mean(if_cofs), np.std(if_cofs),
        np.mean(if_tes), np.std(if_tes)))
        

if __name__ == "__main__":
    main()
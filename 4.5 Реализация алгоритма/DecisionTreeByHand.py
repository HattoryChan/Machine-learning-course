# -*- coding: utf-8 -*-
import numpy as np

'''
Нагло скопировано с https://habr.com/ru/company/mailru/blog/438560/
Но я разобрался как это работает
    (нет, формулу преобразования полностью не вспомню.
     да, знаю что на словах все могут :D  )

Чтобы не читать один и тот же код - напишу что переделано:
    добавлена фукнция расчета MSE по предикту
    комментарии переведены на ломаный английский(потому-что хочу)
    
'''

class RegressionTreeFastMse():
    
    '''
    RegressionTree class with fast eror recalculating. Difficulty of 
      recalculating the error at each iteration - O(1)
    '''
    
    # define class characteristics
    def __init__(self, max_depth=3, min_size=10):
        
        self.max_depth = max_depth
        self.min_size = min_size
        self.value = 0
        self.feature_idx = -1
        self.feature_threshold = 0
        self.left = None
        self.right = None
        self.MSE = []
        
    # learing procedure - we put learning set here
    def fit(self, X, y):
        
        # start value - y mean
        self.value = y.mean()
        # start error - mse between list values (if we doesn't make a split -
        # - it's mean value on all object) and object
        base_error = ((y - self.value) ** 2).sum()
        error = base_error
        
        # we in max deep place
        if self.max_depth <= 1:
            return
    
        dim_shape = X.shape[1]
        
        left_value, right_value = 0, 0
        
        for feat in range(dim_shape):
            
            prev_error1, prev_error2 = base_error, 0 
            idxs = np.argsort(X[:, feat])
            
            # value for fast sum change
            mean1, mean2 = y.mean(), 0
            sm1, sm2 = y.sum(), 0
            
            N = X.shape[0]
            N1, N2 = N, 0
            thres = 1
            
            while thres < N - 1:
                N1 -= 1
                N2 += 1

                idx = idxs[thres]
                x = X[idx, feat]
                
                # calculate deltas - we will use it to change
                delta1 = (sm1 - y[idx]) * 1.0 / N1 - mean1
                delta2 = (sm2 + y[idx]) * 1.0 / N2 - mean2
                
                # sum increase
                sm1 -= y[idx]
                sm2 += y[idx]
                
                # recalculate errors on O(1)
                prev_error1 += (delta1**2) * N1 
                prev_error1 -= (y[idx] - mean1)**2 
                prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1)
                mean1 = sm1/N1
                
                prev_error2 += (delta2**2) * N2 
                prev_error2 += (y[idx] - mean2)**2 
                prev_error2 -= 2 * delta2 * (sm2 - mean2 * N2)
                mean2 = sm2/N2
                
                # skipping near values
                if thres < N - 1 and np.abs(x - X[idxs[thres + 1], feat]) < 1e-5:
                    thres += 1
                    continue
                
                # 2 conditions: make split(minimazed error)
                # and minimal elements count if each leaf
                if (prev_error1 + prev_error2 < error):
                    if (min(N1,N2) > self.min_size):
                    
                        # redefining the better feautures and board along it
                        self.feature_idx, self.feature_threshold = feat, x
                        # redefining the value in leaf
                        left_value, right_value = mean1, mean2

                        
                        error = prev_error1 + prev_error2
                                     
                thres += 1
 
        # if we doesn't have a split - out
        if self.feature_idx == -1:
            return
        
        self.left = RegressionTreeFastMse(self.max_depth - 1)
        # print ("Left subtree with deep value %d"%(self.max_depth - 1))
        self.left.value = left_value
        self.right = RegressionTreeFastMse(self.max_depth - 1)
        # print ("Right subtree with value %d"%(self.max_depth - 1))
        self.right.value = right_value
        
        idxs_l = (X[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (X[:, self.feature_idx] <= self.feature_threshold)
    
        self.left.fit(X[idxs_l, :], y[idxs_l])
        self.right.fit(X[idxs_r, :], y[idxs_r])
        
    def __predict(self, x):
        if self.feature_idx == -1:
            return self.value
        
        if x[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)
        
    def predict(self, X):
        y = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])
            
        return y
    
    def mse(self, y, test, round_ord = 10**2):
        return round(((y - test) ** 2).sum()/len(test), round_ord)
    
    def GridSearchCV(self, model, X_test, y_test, param_grid, cv=10 ):        
        #CV part
        self.MSE.clear()
        cv_point = [0]
        set_len = X_test.size
        cv_point = [ int(set_len/cv)*pn for pn in range(1, cv)]
        cv_point.sort()
        cv_point.append(set_len)
        #print(cv_point)
        
        #to list
        X_test = X_test.tolist()
        y_test = y_test.tolist()
        
        X_test_cv = np.empty
        X_train_cv = np.empty
        y_test_cv = np.empty
        y_train_cv = np.empty
        
        #Spliting the case [---][***][---][---]
        for i in range( len(cv_point), 1, -1):
            X_test_cv = np.array(X_test[i-2:i])
            X_train_cv = np.array(X_test[: X_test.index(X_test[i-2:i][0]) ] +\
                                               X_test[i:])
            
            y_test_cv = np.array(y_test[i-2:i])
            y_train_cv = np.array(y_test[: y_test.index(y_test[i-2:i][0]) ] +\
                                               y_test[i:])
            
        #GridSearch part
        key = next(iter(param_grid[0])) #remember first(start) value
        for key2 in param_grid[0]:
            if key == key2: #exclude the first element
               continue   #we won't repetition
           
            for item in param_grid[0].get(key): #run run run
                for item2 in param_grid[0].get(key2):
                    #Past in model and save output part                    
                    m = model(**{key : item, key2 : item2})
                    #m.fit(X_train_cv, y_train_cv)
                    m.fit(X_train_cv, y_train_cv)
                    #rewrote to dict and create 'best param dict'
                    self.MSE.append(m.mse(m.predict(X_test_cv), y_test_cv))
                    '''
                    #Testing part
                    print(key, item, key2, item2)  #and past in model
                    print(self.MSE[-1])
                    print('\n')
                    '''
            key = key2   
        
        return(self.MSE)
    
    def bestParam():
        return({'MSE': self.MSE})
    
def main():
    import sklearn.datasets as datasets #only for check predict in self run
    
    data = datasets.fetch_california_housing()
    X = np.array(data.data)
    y = np.array(data.target)
    
    '''
    A = RegressionTreeFastMse(4, min_size=7)
    A.fit(X,y)
    test_mytree = A.predict(X)
    predict =  ((y - test_mytree) ** 2).sum()/len(test_mytree)
    A_mse = A.mse(test_mytree, y)  
    print(A_mse, predict)
    '''
    A = RegressionTreeFastMse(4, min_size=7)
    
    param_grid = [{ 'max_depth': range(5),
                     'min_size' : [0,1,2] }]
    A.GridSearchCV(RegressionTreeFastMse, X, y, param_grid, cv=10)
    
if __name__ == "__main__":
    main()
   
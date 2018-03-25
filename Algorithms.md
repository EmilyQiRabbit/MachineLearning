# 算法汇总 之 GridSearch

## Grid Search

通常， Grid Search（网格搜索）可以用来调优参数，但实际上它就是暴力搜索： 
首先，为想要调参的参数设定一组候选值，然后网格搜索就会穷举各种参数组合，根据用户设定的评分规则找到最好的那一组参数。

在 sklearn 算法库中，有专门的[GridSearch算法](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)包，可以直接引用：

```
from sklearn.model_selection import GridSearchCV
```




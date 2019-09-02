import pandas as pd
from sklearn.linear_model import LogisticRegression


class Titanic:

    model = None

    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path)
        return data

    def data_preprocess(self, data, mode='test'):
        # variables selection, normalization
        cnt = data.count()
        data['Sex'] = data['Sex'].replace(['male', 'female'], [1, 0])
        data['Age'] = data['Age'].fillna(0)
        data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
        data['Embarked'] = data['Embarked'].fillna(3)
        data['Fare'] = data['Fare'].fillna(0)
        feature = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        if mode == 'train':
            label = data['Survived']
            return feature, label
        else:
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        data = self.load_data('/Users/lyoung/PycharmProjects/ML/titanic/train.csv')
        feature, label = self.data_preprocess(data, 'train')
        self.model = LogisticRegression(
            solver='liblinear',
            max_iter=100, multi_class='ovr',
            verbose=1
        ).fit(feature, label)

    def test(self):
        data = self.load_data('/Users/lyoung/PycharmProjects/ML/titanic/test.csv')
        feature, p_id = self.data_preprocess(data)
        pre_id = p_id.reset_index(drop=True)
        if self.model is not None:
            pre = self.model.predict(feature)
            prediction = pd.Series(data=pre, name='prediction').to_frame()
            result = pre_id.to_frame().join(prediction)
            result.to_csv(path_or_buf='prediction.csv', index=False)
            return result
        else:
            print 'Model not exists.'
            return


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()

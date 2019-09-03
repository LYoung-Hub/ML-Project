import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = pd.read_csv(path)
    return data


class Titanic:

    model = None
    scaler = StandardScaler()

    def __init__(self):
        pass

    def data_pre_process(self, data, mode='test'):
        # variables selection, normalization
        data['Sex'] = data['Sex'].replace(['male', 'female'], [1, 0])
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
        data['Embarked'] = data['Embarked'].fillna(3)
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        feature = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        if mode == 'train':
            feature = self.scaler.fit_transform(feature)
            label = data['Survived']
            return feature, label
        else:
            feature = self.scaler.transform(feature)
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        data = load_data('/Users/lyoung/PycharmProjects/ML/titanic/train.csv')
        feature, label = self.data_pre_process(data, 'train')
        self.model = LogisticRegression(
            solver='liblinear',
            max_iter=100, multi_class='ovr',
            verbose=1
        ).fit(feature, label)
        acc = self.model.score(feature, label)
        print acc

    def test(self):
        data = load_data('/Users/lyoung/PycharmProjects/ML/titanic/test.csv')
        feature, p_id = self.data_pre_process(data)
        pre_id = p_id.reset_index(drop=True)
        if self.model is not None:
            pre = self.model.predict(feature)
            prediction = pd.Series(data=pre, name='Survived').to_frame()
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

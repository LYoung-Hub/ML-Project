import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def load_data(path):
    data = pd.read_csv(path)
    return data


def modify_csv(path):
    pre = pd.read_csv(path)
    pre['Survived'] = pd.Series(data=pre['Survived'], dtype=int)
    pre.to_csv(path_or_buf='prediction_dnn.csv', index=False)


def cal_ratio():
    data = load_data('titanic/train.csv')
    return data['Survived'].mean()


class Titanic:

    age_scaler = StandardScaler()
    fare_scaler = StandardScaler()

    def __init__(self):
        pass

    def data_pre_process(self, data, mode='test'):
        # variables selection, normalization
        data['Sex'] = data['Sex'].replace(['male', 'female'], [1, 0])
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
        data['Embarked'] = data['Embarked'].fillna(3)
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        feature = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
        if mode == 'train':
            feature[:, 2] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 2], (-1, 1))), (-1))
            feature[:, 5] = np.reshape(self.fare_scaler.fit_transform(np.reshape(feature[:, 5], (-1, 1))), (-1))
            label = data['Survived']
            return feature, label
        else:
            feature[:, 2] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 2], (-1, 1))), (-1))
            feature[:, 5] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 5], (-1, 1))), (-1))
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        data = load_data('titanic/train.csv')
        feature, label = self.data_pre_process(data, 'train')

        # model structure
        input_data = Input(shape=(7, ))
        d = Dense(
            units=128,
            activation='relu'
        )(input_data)

        for i in range(0, 20):
            d = Dense(
                units=128,
                activation='relu'
            )(d)
        d = Dense(
            units=64,
            activation='relu'
        )(d)

        d = Dense(
            units=32,
            activation='relu'
        )(d)

        d = Dense(
            units=16,
            activation='relu'
        )(d)

        d = Dense(
            units=8,
            activation='relu'
        )(d)

        d = Dense(
            units=4,
            activation='relu'
        )(d)

        output = Dense(
            units=1,
            activation='sigmoid'
        )(d)

        model = Model(input_data, output)
        model.summary()

        adam = Adam(lr=0.0001)
        model.compile(
            optimizer=adam,
            loss='mse',
            metrics=['accuracy']
        )

        # callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='models/best_model.hdf5',
                save_best_only=True,
                verbose=1,
                period=1,
                mode='auto',
                monitor='val_loss'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=10,
                factor=0.1,
                mode='auto',
                min_lr=0.0,
                verbose=1
            )
        ]

        model.fit(
            x=feature,
            y=label,
            batch_size=1,
            epochs=10000,
            shuffle=True,
            validation_split=0.3,
            callbacks=callbacks,
            verbose=1
        )

        model.save('model_dnn.hdf5')

    def test(self):
        data = load_data('titanic/test.csv')
        feature, p_id = self.data_pre_process(data)
        pre_id = p_id.reset_index(drop=True)

        model = load_model('models/best_model.hdf5')
        pre = model.predict(feature)

        # change to acceptable result
        pre = np.reshape(pre, 418)
        ratio = cal_ratio()
        for i in range(0, 418):
            if pre[i] < ratio:
                pre[i] = 0
                pre[i] = int(pre[i])
            else:
                pre[i] = 1
                pre[i] = int(pre[i])

        # create csv
        prediction = pd.Series(data=pre, name='Survived').to_frame()
        result = pre_id.to_frame().join(prediction)
        result.to_csv(path_or_buf='prediction_dnn.csv', index=False)
        return result


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
    modify_csv('prediction_dnn.csv')

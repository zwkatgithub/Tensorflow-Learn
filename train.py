import os
from flags import FLAGS
from model import UserModel, UserModel2
from dataloader import UserDataLoader
from framework import UserFramework

train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train.xlsx')
test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test.xlsx')

train_data_loader = UserDataLoader(train_file, FLAGS.batch_size, shuffle=True)
test_data_loader = UserDataLoader(test_file, FLAGS.batch_size, shuffle=False)
#print(next(test_data_loader))

user_model = UserModel()
user_model2 = UserModel2()

user_framework = UserFramework(train_data_loader, test_data_loader)

#user_framework.build(user_model, 'model1')
user_framework.train(user_model2, prefix='model2')

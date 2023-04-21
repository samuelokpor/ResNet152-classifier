from main import res152instance
#from fuionresnet50 import fusedresnetinstance
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.callbacks import CSVLogger, ModelCheckpoint
import pandas as pd
from dataset import train, val

res152instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), Precision(), Recall()])
res152instance.summary() 
#define callbacks
logdir = 'logs'
csv_logger = CSVLogger('training_history.csv')
checkpoint = ModelCheckpoint('model2.h5', save_best_only=True)

# #Train the Model
history = res152instance.fit(train, epochs=50, validation_data=val, callbacks=[csv_logger, checkpoint])

#save the history in a datframe
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history_df.csv', index=False)

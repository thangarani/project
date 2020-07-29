Text Summarization Pegasus:
==========================
INPUTS: [0]:
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow). At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments. WHO will continue to provide updated information as soon as clinical findings become available.
TARGETS: Coronavirus disease is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. However, there are many ongoing clinical trials evaluating potential treatments.
PREDICTIONS: Coronavirus disease

I made some changes in the code to create our own input data. Please follow the below instruction to make it work.

For Selected input and target:
=============================
The input needs to be a .tfrecord. So let’s just see how we are going to create our input data. The following piece of code ought to do it for you. Just one thing to take care of here, make sure the .tfrecord is saved inside the testdata directory, which is inside pegasus/data/.

Path: /content/pegasus/create_input_file.py
============================================
import pandas as pd
import tensorflow as tf

save_path = "/content/pegasus/data/testdata/test_pattern_1.tfrecord"

input_dict = dict(
                  inputs=[
                          "Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow). At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments. WHO will continue to provide updated information as soon as clinical findings become available."
                         ],
                  targets=[
                          "Coronavirus disease is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special  treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. However, there are many ongoing clinical trials evaluating potential treatments."
                          ]
                 )

data = pd.DataFrame(input_dict)

with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())
 
In the gist above you will see that the targets are also passed. The list target is supposed to be the actual summary or the ground truth. Since we are only trying to generate summaries from the model and not train it, you can pass empty strings, but we can’t omit it because the model expects input in that format.
Awesome! Now that our data is prepared, there is just one more step we start to get the summaries. So this step is to register our tfrecord in the registry of the pegasus(locally). Great! Let’s move forward.

Path: /content/pegasus/params/public_params.py
==============================================
@registry.register("new_params")
def new_params(param_overrides):
  return transformer_params(
    {
          "train_pattern": "tfrecord:/content/pegasus/data/testdata/test_pattern_1.tfrecord",
          "dev_pattern": "tfrecord:/content/pegasus/data/testdata/test_pattern_1.tfrecord",
          "test_pattern": "tfrecord:/content/pegasus/data/testdata/test_pattern_1.tfrecord",
          "max_input_len": 512,
          "max_output_len": 32,
          "train_steps": 32000,
          "learning_rate": 0.0001,
          "batch_size": 8,
    }, param_overrides)
    
In the pegasus directory in your system, go to the path pegasus/params/public_params.py and paste the above code at the end of the script. In the above gist you will see that all the three; train_pattern, dev_pattern and test_pattern are assigned the same tfrecord, you may create different tfrecords for all three but since we are only looking to infer, it doesn’t matter. And we are done!
    
!python3 pegasus/bin/evaluate.py --params=new_params \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=/content/cnn_dailymail
    

This will start to create your summaries for your input data. Once done you will see 3 text files created in the directory of the model that you pick. These three files correspond to the input text, target text and the predicted summaries. You can open these text files and analyze the summaries.

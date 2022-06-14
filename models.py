from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, BatchNormalization, TimeDistributed, LSTM, concatenate

from constants import EMOTIONS


# test
# ****************************** RL MODELS ******************************


def get_model_9_rl(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(bn1)

    f1 = TimeDistributed(Flatten())(c2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='linear', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9')

    return model


# ****************************** MULTI INPUT MODELS ******************************


def get_model_9_2_multi_input(input_layers, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layers[0])
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=4, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    # f1 = TimeDistributed(Flatten())(mp2)
    # lstm = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(mp2)

    # emotion part
    emo_d1 = Dense(320, activation='relu')(f1)

    mfcc = Model(inputs=input_layers[0], outputs=emo_d1)

    egeMaps_d1 = Dense(200)(input_layers[1])
    egeMaps_d2 = Dense(160, activation='relu')(egeMaps_d1)
    egeMaps_d3 = Dense(80, activation='relu')(egeMaps_d2)

    egemaps = Model(inputs=input_layers[1], outputs=egeMaps_d3)

    combined = concatenate([mfcc.output, egemaps.output])

    combined_d2 = Dense(200, activation='relu')(combined)
    combined_d3 = Dense(128, activation='relu')(combined_d2)

    combined_dr1 = Dropout(0.3)(combined_d3)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_9_2_multi_input')

    return model


def get_model_12_multi_input(input_layers, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=16, padding='same', activation='elu')(input_layers[0])
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=4)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm1 = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm1)

    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    mfcc = Model(inputs=input_layers[0], outputs=emo_d2)

    egeMaps_d1 = Dense(200)(input_layers[1])
    egeMaps_d2 = Dense(160, activation='relu')(egeMaps_d1)
    egeMaps_d3 = Dense(80, activation='relu')(egeMaps_d2)

    egemaps = Model(inputs=input_layers[1], outputs=egeMaps_d3)

    combined = concatenate([mfcc.output, egemaps.output])

    combined_d2 = Dense(200, activation='relu')(combined)
    combined_d3 = Dense(128, activation='relu')(combined_d2)

    combined_dr1 = Dropout(0.3)(combined_d3)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_12_multi_input')

    return model


def get_model_12_1_multi_input(input_layers, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=16, padding='same', activation='elu')(input_layers[0])
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=4)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm1 = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm1)

    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    mfcc = Model(inputs=input_layers[0], outputs=emo_d2)

    egeMaps_1_d1 = Dense(200)(input_layers[1])
    egeMaps_1_d2 = Dense(160, activation='relu')(egeMaps_1_d1)
    egeMaps_1_d3 = Dense(80, activation='relu')(egeMaps_1_d2)

    egemaps_1 = Model(inputs=input_layers[1], outputs=egeMaps_1_d3)

    egeMaps_2_d1 = Dense(200)(input_layers[2])
    egeMaps_2_d2 = Dense(160, activation='relu')(egeMaps_2_d1)
    egeMaps_2_d3 = Dense(80, activation='relu')(egeMaps_2_d2)

    egemaps_2 = Model(inputs=input_layers[2], outputs=egeMaps_2_d3)

    egeMaps_3_d1 = Dense(200)(input_layers[3])
    egeMaps_3_d2 = Dense(160, activation='relu')(egeMaps_3_d1)
    egeMaps_3_d3 = Dense(80, activation='relu')(egeMaps_3_d2)

    egemaps_3 = Model(inputs=input_layers[3], outputs=egeMaps_3_d3)

    combined = concatenate([mfcc.output, egemaps_1.output, egemaps_2.output, egemaps_3.output])

    combined_d2 = Dense(200, activation='relu')(combined)
    combined_d3 = Dense(128, activation='relu')(combined_d2)

    combined_dr1 = Dropout(0.3)(combined_d3)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_12_1_multi_input')

    return model


def get_model_17(input_layers, model_name_prefix=''):
    egeMaps_1_d1 = Dense(200)(input_layers[0])
    egeMaps_1_d2 = Dense(160, activation='relu')(egeMaps_1_d1)
    egeMaps_1_d3 = Dense(80, activation='relu')(egeMaps_1_d2)

    egemaps_1 = Model(inputs=input_layers[0], outputs=egeMaps_1_d3)

    egeMaps_2_d1 = Dense(200)(input_layers[1])
    egeMaps_2_d2 = Dense(160, activation='relu')(egeMaps_2_d1)
    egeMaps_2_d3 = Dense(80, activation='relu')(egeMaps_2_d2)

    egemaps_2 = Model(inputs=input_layers[1], outputs=egeMaps_2_d3)

    combined = concatenate([egemaps_1.output, egemaps_2.output])

    combined_d2 = Dense(200, activation='relu')(combined)
    combined_d3 = Dense(128, activation='relu')(combined_d2)

    combined_dr1 = Dropout(0.3)(combined_d3)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_17')

    return model


def get_model_18_multi_input(input_layers, model_name_prefix=''):
    egeMaps_1_d1 = Dense(200)(input_layers[0])
    egeMaps_1_d2 = Dense(160, activation='relu')(egeMaps_1_d1)
    egeMaps_1_d3 = Dense(80, activation='relu')(egeMaps_1_d2)

    egemaps_1 = Model(inputs=input_layers[0], outputs=egeMaps_1_d3)

    egeMaps_2_d1 = Dense(200)(input_layers[1])
    egeMaps_2_d2 = Dense(160, activation='relu')(egeMaps_2_d1)
    egeMaps_2_d3 = Dense(80, activation='relu')(egeMaps_2_d2)

    egemaps_2 = Model(inputs=input_layers[1], outputs=egeMaps_2_d3)

    egeMaps_3_d1 = Dense(200)(input_layers[2])
    egeMaps_3_d2 = Dense(160, activation='relu')(egeMaps_3_d1)
    egeMaps_3_d3 = Dense(80, activation='relu')(egeMaps_3_d2)

    egemaps_3 = Model(inputs=input_layers[2], outputs=egeMaps_3_d3)

    combined = concatenate([egemaps_1.output, egemaps_2.output, egemaps_3.output])

    combined_d2 = Dense(200, activation='relu')(egeMaps_3_d3)
    combined_d3 = Dense(128, activation='relu')(combined_d2)

    combined_dr1 = Dropout(0.3)(combined_d3)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_18_multi_input')

    return model


def get_model_19_multi_input(input_layers, model_name_prefix=''):
    def _flb(input, d1, d2, d3):
        egeMaps_d1 = Dense(d1)(input)
        egeMaps_d2 = Dense(d2, activation='relu')(egeMaps_d1)
        egeMaps_d3 = Dense(d3, activation='relu')(egeMaps_d2)
        return egeMaps_d3

    c1 = Conv2D(7, kernel_size=16, padding='same', activation='elu')(input_layers[0])
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=4)(bn1)
    f1 = TimeDistributed(Flatten())(mp1)
    lstm1 = LSTM(32, return_sequences=True)(f1)
    f1 = Flatten()(lstm1)
    emo_d1 = Dense(256, activation='relu')(f1)
    emo_d2 = Dense(128, activation='relu')(emo_d1)
    mfcc = Model(inputs=input_layers[0], outputs=emo_d2)

    egeMaps_pitch_d3 = _flb(input_layers[1], 200, 160, 80)
    egeMaps_pitch = Model(inputs=input_layers[1], outputs=egeMaps_pitch_d3)

    egeMaps_formantF1freq_d3 = _flb(input_layers[2], 200, 160, 80)
    egeMaps_formantF1freq = Model(inputs=input_layers[2], outputs=egeMaps_formantF1freq_d3)

    egeMaps_formantF2freq_d3 = _flb(input_layers[3], 200, 160, 80)
    egeMaps_formantF2freq = Model(inputs=input_layers[3], outputs=egeMaps_formantF2freq_d3)

    egeMaps_formantF3freq_d3 = _flb(input_layers[4], 200, 160, 80)
    egeMaps_formantF3freq = Model(inputs=input_layers[4], outputs=egeMaps_formantF3freq_d3)

    egeMaps_loudness_d3 = _flb(input_layers[5], 200, 160, 80)
    egeMaps_loudness = Model(inputs=input_layers[5], outputs=egeMaps_loudness_d3)

    egeMaps_HNR_d3 = _flb(input_layers[6], 200, 160, 80)
    egeMaps_HNR = Model(inputs=input_layers[6], outputs=egeMaps_HNR_d3)

    egeMaps_slope0_d3 = _flb(input_layers[7], 200, 160, 80)
    egeMaps_slope0 = Model(inputs=input_layers[7], outputs=egeMaps_slope0_d3)

    egeMaps_slope500_d3 = _flb(input_layers[8], 200, 160, 80)
    egeMaps_slope500 = Model(inputs=input_layers[8], outputs=egeMaps_slope500_d3)

    combined = concatenate([mfcc.output,
                            egeMaps_pitch.output,
                            egeMaps_formantF1freq.output,
                            egeMaps_formantF2freq.output,
                            egeMaps_formantF3freq.output,
                            egeMaps_loudness.output,
                            egeMaps_HNR.output,
                            egeMaps_slope0.output,
                            egeMaps_slope500.output,
                            ])

    combined_d2 = Dense(500, activation='relu')(combined)
    combined_d3 = Dense(200, activation='relu')(combined_d2)
    combined_d4 = Dense(64, activation='relu')(combined_d3)

    combined_dr1 = Dropout(0.3)(combined_d4)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_dr1)

    model = Model(inputs=input_layers, outputs=[d_out], name=model_name_prefix + '_model_19_multi_input')

    return model


def get_model_20_multi_input(input_layers, model_name_prefix=''):
    mfcc_c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layers[0])
    mfcc_bn1 = BatchNormalization()(mfcc_c1)
    mfcc_mp1 = MaxPooling2D(strides=2)(mfcc_bn1)
    mfcc_c2 = Conv2D(3, kernel_size=4, padding='same', activation='relu')(mfcc_mp1)
    mfcc_mp2 = MaxPooling2D(strides=2)(mfcc_c2)
    mfcc_f1 = TimeDistributed(Flatten())(mfcc_mp2)
    mfcc_lstm = LSTM(32, return_sequences=True)(mfcc_f1)
    mfcc_f1 = Flatten()(mfcc_lstm)
    mfcc_emo_d1 = Dense(320, activation='relu')(mfcc_f1)
    mfcc = Model(inputs=input_layers[0], outputs=mfcc_emo_d1)

    egemaps_c1 = Conv2D(32, kernel_size=8, padding='same', activation='relu')(input_layers[1])
    egemaps_bn1 = BatchNormalization()(egemaps_c1)
    egemaps_mp1 = MaxPooling2D(strides=2)(egemaps_bn1)
    egemaps_c2 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(egemaps_mp1)
    egemaps_mp2 = MaxPooling2D(strides=2)(egemaps_c2)
    egemaps_f1 = TimeDistributed(Flatten())(egemaps_mp2)
    egemaps_lstm = LSTM(16, return_sequences=True)(egemaps_f1)
    egemaps_f1 = Flatten()(egemaps_lstm)
    egemaps_emo_d1 = Dense(320, activation='relu')(egemaps_f1)
    egemaps_dr = Dropout(0.7)(egemaps_emo_d1)
    egemaps = Model(inputs=input_layers[1], outputs=egemaps_dr)

    combined_1 = concatenate([mfcc.output, egemaps.output])

    combined_1_d2 = Dense(500, activation='relu')(combined_1)
    combined_1_dr1 = Dropout(0.7)(combined_1_d2)
    combined_1_d3 = Dense(320, activation='relu')(combined_1_dr1)
    combined_1_d4 = Dense(128, activation='relu')(combined_1_d3)

    emo_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_1_d4)

    model = Model(inputs=input_layers, outputs=emo_out, name=model_name_prefix + '_model_20_multi_input')

    return model


def get_model_21_multi_input(input_layers, model_name_prefix=''):
    mfcc_c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layers[0])
    mfcc_bn1 = BatchNormalization()(mfcc_c1)
    mfcc_mp1 = MaxPooling2D(strides=2)(mfcc_bn1)
    mfcc_f1 = TimeDistributed(Flatten())(mfcc_mp1)
    mfcc_lstm = LSTM(32, return_sequences=True)(mfcc_f1)
    mfcc_f1 = Flatten()(mfcc_lstm)
    mfcc_emo_d1 = Dense(320, activation='relu')(mfcc_f1)
    mfcc = Model(inputs=input_layers[0], outputs=mfcc_emo_d1)

    egemaps_c1 = Conv2D(7, kernel_size=8, padding='same', activation='relu')(input_layers[1])
    egemaps_bn1 = BatchNormalization()(egemaps_c1)
    egemaps_mp1 = MaxPooling2D(strides=2)(egemaps_bn1)
    egemaps_f1 = TimeDistributed(Flatten())(egemaps_mp1)
    egemaps_lstm = LSTM(8, return_sequences=True)(egemaps_f1)
    egemaps_f1 = Flatten()(egemaps_lstm)
    egemaps_emo_d1 = Dense(160, activation='relu')(egemaps_f1)
    egemaps_dr = Dropout(0.7)(egemaps_emo_d1)
    egemaps = Model(inputs=input_layers[1], outputs=egemaps_dr)

    combined_1 = concatenate([mfcc.output, egemaps.output])

    combined_1_d2 = Dense(500, activation='relu')(combined_1)
    combined_1_dr1 = Dropout(0.7)(combined_1_d2)
    combined_1_d3 = Dense(320, activation='relu')(combined_1_dr1)
    combined_1_d4 = Dense(128, activation='relu')(combined_1_d3)

    emo_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(combined_1_d4)

    model = Model(inputs=input_layers, outputs=emo_out, name=model_name_prefix + '_model_21_multi_input')

    return model

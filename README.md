# NMT-LSTM-EN_TR
- This model is acquired and changed from DeepLearning.AI - Week 1 (Neural Machine Translation) course assignment-1.
 
Implementing an encoder-decoder system with attention (Only RNN with LSTMs not the best for long sentences). Building the NMT model from scratch using Tensorflow. Generating translations using greedy and Minimum Bayes Risk (MBR) decoding

# RESULTS

English (to translate) sentence:
You have the same camera as mine.

Turkish (translation) sentence:
Sen benimki ile aynı kameraya sahipsin.

First 10 words of the english vocabulary:
['', '[UNK]', '[SOS]', '[EOS]', '.', 'the', 'you', 'to', 'is', 'a']

First 10 words of the turkish vocabulary:
['', '[UNK]', '[SOS]', '[EOS]', '.', 'bir', '?', 'bu', ',', 'o']

Turkish vocabulary is made up of 12000 words
The id for the [UNK] token is 1
The id for the [SOS] token is 2
The id for the [EOS] token is 3
The id for baunilha (vanilla) is 1

Tokenized english sentence:
[   2   94   26  681 1273   40 4567   15   32 4744 4751   11    3    0    0    0    0    0    0]


Tokenized turkish sentence (shifted to the right):
[   2  250   72 1974   38    1    8  201   76  151 9880    6    0    0    0]


Tokenized turkish sentence:
[ 250   72 1974   38    1    8  201   76  151 9880    6    3    0    0    0]


Tensor of sentences in english has shape: (64, 19)
Encoder output has shape: (64, 19, 256)

Tensor of contexts has shape: (64, 19, 256)
Tensor of translations has shape: (64, 15, 256)
Tensor of attention scores has shape: (64, 15, 256)

Tensor of contexts has shape: (64, 19, 256)
Tensor of right-shifted translations has shape: (64, 15)
Tensor of logits has shape: (64, 15, 12000)

Tensor of sentences to translate has shape: (64, 19)
Tensor of right-shifted translations has shape: (64, 15)
Tensor of logits has shape: (64, 15, 12000)


Model: "translator"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 encoder_11 (Encoder)        multiple                  4122624   
                                                                 
 decoder_11 (Decoder)        multiple                  7470304   
                                                                 
=================================================================
Total params: 11,592,928
Trainable params: 11,592,928
Non-trainable params: 0
_________________________________________________________________
Next token: [[29]]
Logit: -6.7207
Done? False
Temperature: 0.0

Original sentence: Whats up bro
Translation: gor kokteyl ayrlmak yapmadn uretildi parlyordu yontemler sevgililer yapmamn yapmamn oldugunuz oldugunuz oldugunuz oldugunuz yaygara yaygara koymayn yaygara sarkyla genis yaygara genis yaygara oynuyorsunuz duyabiliyor edilir bari ayrldgn leke cekicidir ediyorlard ediyorlard sallama birden birden yelkenli planmz arabanz ozetleyelim verecegine orijinali duymustum temizleyiciye temizleyiciye raket harcar harcar kanser soydu goruyorum
Translation tokens:[[ 2394  6593  1634  8949  9492 11732  8472  2649  8934  8934  4237  4237
   4237  4237  3084  3084  3373  3084 11153  3540  3084  3540  3084 11830
   4832  2928  7964  5098  3358  7716  7411  7411 11235  3705  3705  5332
   6086  3773 11808  9227 11939  7432  9923  9923  4160  7021  7021  1746
   2630  2875]]
Logit: -9.362
Temperature: 0.3

Original sentence: Whats up bro
Translation: uyuyamadn meshurdur tatsuyann ckan tavr portekizceyi direkt tehlikeyle yardmda rezervasyonlar nefese ogretti adamsn gelmistir tasr toplamak snrsn etkilendim sirketindeki sistemimiz varm uyesi politikacdr bulusmalyz volkan yarsn rahatlatc patlamasna skor tanmyorduk arkadasmla ulkesidir gazeteyi tehdit oranlar hayatmdaki trang habersiz willin jon yonleriyle romanlarnn surunen alsaydm odan iflas yasama piyanoda anlamyorum yandaki
Translation tokens:[[ 5512  6365 10026  7672 10017 11549  7528  9980  8841 11406  6319  6265
   5203  7218  3151  5621 10669  2918 10777 10753  9283  1899 11563  7784
   2148  5375 11484 11700  5864 10173  1846  3125  7265  5678 11948  6986
   9743  7036  9125  6785  8476  6039 10315  8138  2731  6914  3882 11603
   1096  9021]]
Logit: -31.267
Temperature: 0.7

Original sentence: Whats up bro
Translation: kulaga tadma elektrikle sorabilir olmayacak satarlar jackin hoslanacagnz otobusunun renk erteledik ykanmal ugurlamak seyleri yoshida yiyor farknda biliniyordu gitmeyin onlarnkinden saglgnz semboludur kaynaklar secildi yagsl evidir krmz skandal safakta yapn oluyor kl hokkaidoda isin transfer kardesin prens sahradr yarallar zayflara trenden cicek sevindim ulkenize kostum cocuklarla gelmeyi fil bocekleri pasaporta
Translation tokens:[[ 4379 10259  7391  1694  2236  3233  2331  6949 11892  2672  7370  8548
   9585  2645  5267  3850  4761  7878  4686 11975  3245  3219  4470  3224
   2577  7329   816  3202 11314  1666   597  4432  6953  1150  9741  1745
   2677 11276  8874  8298  9739   558  2648  9558  2289  7651  7220  4748
   7830  3280]]
Logit: -13.412
Temperature: 1.0

Original sentence: Whats up bro
Translation: yazmaktr dagtmak politikacdr yasarlar posterler zevkimdir seri balg problemlerini universiteyi dunyann verecegim bizimle hayal hatayd yaptgnz uzag uzgunuz tu ulkelerine yas parcaland haksz sirketleri kuvvetli secin varsaylr vardr ulkeye sacmalama gezisine bitirmenin yoksullara sark sekildeydi yakalanabilirsiniz donarak kendisinin olsan yitirmisti pazarlamak sarf geziye topraga vazgecebilsem ettigim gordugun yastasn varsaylr tanst
Translation tokens:[[ 8689  7621 11563  2133 11538  8275  5930  1631  4167  9513   727   519
    835   659  4632  8901  9358  5497  9718  9563  3885  6131  7031 10771
   2760  5948  9273   116  2161 11327  4705  7841  5288   263 11004  9083
   7497  4453  6213  8566 11673 11162  4704  9779  9260  1425  7139  8769
   9273  5718]]
Logit: -9.395

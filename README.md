# NMT-LSTM-EN_TR
- This model is acquired and changed from DeepLearning.AI - Week 1 (Neural Machine Translation) course assignment-1.
 
Implementing an encoder-decoder system with attention (Only RNN with LSTMs not the best for long sentences). Building the NMT model from scratch using Tensorflow. Generating translations using greedy and Minimum Bayes Risk (MBR) decoding

**Temperature:** 0.0
**Original sentence:** Whats up bro

**Translation: ** gor kokteyl ayrlmak yapmadn uretildi parlyordu yontemler sevgililer yapmamn yapmamn oldugunuz oldugunuz oldugunuz oldugunuz yaygara yaygara koymayn yaygara sarkyla genis yaygara genis yaygara oynuyorsunuz duyabiliyor edilir bari ayrldgn leke cekicidir ediyorlard ediyorlard sallama birden birden yelkenli planmz arabanz ozetleyelim verecegine orijinali duymustum temizleyiciye temizleyiciye raket harcar harcar kanser soydu goruyorum

**Translation tokens:**[[ 2394  6593  1634  8949  9492 11732  8472  2649  8934  8934  4237  4237   4237  4237  3084  3084  3373  3084 11153  3540  3084  3540  3084 11830   4832  2928  7964  5098  3358  7716  7411  7411 11235  3705  3705  5332   6086  3773 11808  9227 11939  7432  9923  9923  4160  7021  7021  1746   2630  2875]]
**Logit:** -9.362
===============================================================================
**Temperature:** 0.7
**Original sentence:** Whats up bro

**Translation: **kulaga tadma elektrikle sorabilir olmayacak satarlar jackin hoslanacagnz otobusunun renk erteledik ykanmal ugurlamak seyleri yoshida yiyor farknda biliniyordu gitmeyin onlarnkinden saglgnz semboludur kaynaklar secildi yagsl evidir krmz skandal safakta yapn oluyor kl hokkaidoda isin transfer kardesin prens sahradr yarallar zayflara trenden cicek sevindim ulkenize kostum cocuklarla gelmeyi fil bocekleri pasaporta

**Translation tokens:**[[ 4379 10259  7391  1694  2236  3233  2331  6949 11892  2672  7370  8548   9585  2645  5267  3850  4761  7878  4686 11975  3245  3219  4470  3224   2577  7329   816  3202 11314  1666   597  4432  6953  1150  9741  1745   2677 11276  8874  8298  9739   558  2648  9558  2289  7651  7220  4748   7830  3280]]
**Logit:** -13.412

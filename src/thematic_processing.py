import numpy as np
# packages needed for the TF-IDF
# patch_sklearn is an Intel® patch that allows to optimize sklearn operations
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_extraction.text import TfidfVectorizer

# packages needed for LDA grouping
from gensim import corpora, models

# packages needed for lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def lemmatize_docs(docs):
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_docs = []
    # iterate through all the docs to lemmatize them
    for doc in docs:
        # tokenize the docs
        sentence = []
        # iterate through all the words in one document
        for word, tag in pos_tag(word_tokenize(doc)):
            # find the tag for the part of speech
            # v : verb, n : noun,...
            wntag = tag[0].lower()
            # only allowed tags are a,r,n,v
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            # if there is no tag we don't do any changes
            if not wntag:
                lemma = word
            # otherwise we lemmatize
            else:
                lemma = lemmatizer.lemmatize(word, wntag)
            sentence.append(lemma)
        # then we detokenize the sentence, and add it to the lemmatized doc
        sentence = TreebankWordDetokenizer().detokenize(sentence)
        lemmatized_docs.append(sentence)
    return lemmatized_docs


def words_cleaner(words, min_size = 0, del_nums = True):
    """
    This function takes a list of words, and cleans it based on the parameters provided.
    
    Parameters
    ----------
    words : list
        List of words to clean.
    min_size : int
        All words with size < min_size are removed.
    del_nums : bool
        If True, removes all words which can be converted to numbers.
        
    Returns
    -------
    words_clean : list
        List with the clean words obeying the parameters provided
    """
    words_clean = []
    for word in words:
        if del_nums:
            # if we want to remove numbers, then we try to convert the word to number
            try:
                float(word)
            # if it raises an exception then it's fine
            except:
                pass
            # if no exceptions are raised then we skip it
            else:
                continue
        
        # we also only want words bigger than min_size
        if len(word) >= min_size:
            words_clean.append(word)
            
    # return clean word list
    return words_clean

def keywords_extractor_sklearn(docs, num_words = 100, min_size = 0):
    """
    This function takes a series of documents and extracts the 'num_words'
    most important keywords from it, using TF-IDF to spot these words.
    
    Parameters
    ----------
    docs : list
        List of str, contains the docs we wish to analyze.
    num_words : int
        Number of words to return.
    min_size : int
        All words with len(word) < min_size are automatically excluded from
        the output. If None does nothing.
        
    Returns
    -------
    top_words : list
        List containing the most relevant keywords from the docs.
    """
    
    # Generate a vectorizer to analyze the documents via TF-IDF
    # Automatically strip word accents if possible and exclude all stop_words in english language
    vectorizer = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english')
    # Get the TF-IDF Matrix from the docs provided
    tf_idf_matrix = vectorizer.fit_transform(docs)
    # And get the words from the same docs
    words = np.array(vectorizer.get_feature_names_out())
    # Then sort the words according to their TF-IDF weights
    tfidf_sorting = np.argsort(tf_idf_matrix.toarray()).flatten()[::-1]
    
    # Clean the words and return only the most relevant
    top_words = words_cleaner(words[tfidf_sorting], min_size = min_size)[:num_words]
    
    return top_words

def topic_cluster(words, docs):
    """
    This function takes a set of words and clusters them into topics.
    
    
    This function takes the most important words retrieved from a series of
    documents and clusters them into topics.
    
    Parameters
    ----------
    words : list
        List of words that we wish to cluster into categories.
        
    Returns
    -------
    ???
    """
    list_of_list_of_tokens = []
    for doc in docs:
        list_of_tokens = []
        for word in words:
            if word in doc:
                list_of_tokens.append(word)
        list_of_list_of_tokens.append(list_of_tokens)
        
    dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
    
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

    num_topics = 3
    
    lda_model = models.LdaModel(corpus,
                                num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=10,
                                alpha=[0.01]*num_topics,
                                eta=[0.01]*len(dictionary_LDA.keys())
                               )
    
    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=5):
        print(str(i)+": "+ topic)
        print()
        
    for corpse in corpus:
        print(lda_model[corpse])

    
    
if __name__ == '__main__':
    bitcoin_news = ['''As the end of the year approaches, the price of bitcoin has hovered above the $60K region and with 18.8 million bitcoin in circulation, bitcoin’s market valuation is over $1.16 trillion today. Meanwhile, the number of tokenized bitcoins in existence today has swelled significantly during the last three years, climbing to 408,210 bitcoin worth $25 billion today. More Than 400,000 Wrapped, Synthetic, or Tokenized Bitcoins in the Wild. Wrapped, synthetic, or tokenized bitcoin has become a growing trend during the last two years and nine months. Bitcoin.com News reported on one of the first projects on January 30, 2019, the day the Wrapped Bitcoin (WBTC) project first launched. Since then, there have been a whole lot more tokenized bitcoin projects and by July 2019, WBTC in circulation eclipsed the Lightning Network capacity. Now there’s a slew of tokenized bitcoin projects such as BEP2, HBTC, RENBTC, SBTC, PBTC, OBTC, TBTC, Mstablebtc, RBTC, and LBTC. Out of all the aforementioned tokenized bitcoin protocols including WBTC, there are approximately 408,210 tokenized bitcoins in circulation worth $25 billion today. WBTC holds the lion’s share of tokenized BTC with 231,659 tokens in circulation today. The coin BEP2, otherwise known as BTCB issued by Binance, has around 105,099 tokens circulating today. Meanwhile, the other tokenized bitcoin projects have much lower supplies, and the third-largest tokenized BTC project is backed by the trading platform Huobi. There’s 39,884 HBTC (Huobi BTC) today and the valuation of the entire HBTC market is $2.4 billion. HBTC is followed by RENBTC (16,818), SBTC (4,775), LBTC (3,367), RBTC (2,528), PBTC (1,786), OBTC (1,254), TBTC (792), Mstablebtc (248), respectively. Ethereum, Binance Smart Chain Command Lion’s Share of Tokenized Bitcoin. Ethereum is the largest blockchain in terms of the amount of tokenized BTC leveraged on a network. Eight out of the 11 projects that issue wrapped, synthetic, or tokenized bitcoin products use the Ethereum chain. BEP2 (BTCB) stems from the Binance Smart Chain (BSC), RBTC is issued by the RSK network, and LBTC is issued by Blockstream’s Liquid network. The two tokenized BTC projects that have seen exponential growth since launching are WBTC and BEP2. Both projects are the most dominant with an aggregate of 336,758 tokenized bitcoins or 82.49% of all the tokenized BTC in existence.''',
                   '''Bitcoin futures open interest continues to remain high after the launch of the first bitcoin exchange-traded fund (ETF) on October 22. While Binance commands $5.81 billion and leads the pack, CME Group holds the second-largest position in terms of bitcoin futures open interest (OI) with $4.1 billion or 16.84% of the aggregate OI. Top Ten Derivatives Platforms Command More Than 97% of Bitcoin Futures Open Interest. Bitcoin derivatives have swelled quite a bit in recent times, and following the launch of the Proshares and Valkyrie bitcoin futures ETFs, bitcoin futures have seen significant market action. After Valkyrie’s ETF (Nasdaq:BTF) launched, the analytics group Skew tweeted that CME Group’s bitcoin futures OI reached a new all-time high. During the first week of November, bitcoin futures open interest across all the crypto derivatives platforms is $24.32 billion according to coinglass.com statistics. The top ten derivatives platforms offering bitcoin futures command $23.68 billion or more than 97% of the OI. The cryptocurrency trading platform Binance is the leader in terms of bitcoin futures with $73 billion in global volume among 54 different crypto markets. CME Group’s Bitcoin Futures Open Interest More Than 16% of Aggregate OI. In terms of bitcoin futures, Binance holds 5.81 billion in OI which accounts for 23.9% of all the BTC futures positions. Meanwhile, the world’s largest financial derivatives exchange, Chicago Mercantile Exchange (CME) Group, captures 16.84% or $4.1 billion in open interest. Leaving out Binance and CME Group, the top ten crypto derivatives platforms in terms of bitcoin futures OI include exchanges such as FTX ($3.84B), Bybit ($3.63B), Okex ($2.21B), Deribit ($1.49B), Bitfinex ($827.71M), Bitmex ($752.43M), Bitget ($557.5M), and Huobi ($485.59M). Bitcoin ETF Markets Follow Spot Market Trends. In terms of crypto derivatives, 627 crypto futures, and perpetuals across the board, FTX holds the second-largest global volume below Binance with $13.4 billion in 24 hours. Furthermore, much like bitcoin’s (BTC) spot market consolidation period, the bitcoin futures ETFs from Proshares (NYSE:BITO) and Valkyrie have followed similar paths. While BITO swapped at a high of $43.28 in October, shares are currently swapping for $39.30. The exchange-traded fund BTF hit a high of $25.25 but is now changing hands for $24.23.''',
                   '''Paytm, one of India’s largest payments companies, is open to offering bitcoin services if the crypto asset becomes legal in the country, according to its chief financial officer. If bitcoin “was ever to become fully legal in the country, then clearly there could be offerings we could launch,” he said. Paytm Open to Bitcoin Offerings. Paytm Chief Financial Officer Madhur Deora has indicated that his company is open to offering bitcoin services if the crypto asset becomes legal in India, local media reported Thursday, citing his recent interview with Bloomberg TV. Deora was quoted as saying: Bitcoin is still in a regulatory grey area if not a regulatory ban in India. At the moment Paytm does not do bitcoin. If it was ever to become fully legal in the country, then clearly there could be offerings we could launch. In August last year, Paytm reportedly froze Paytm Payments Bank’s customer accounts suspected of crypto trading. Paytm is currently India’s second most-valuable internet company. The company is planning to launch an initial public offering (IPO) between Nov. 8 and Nov. 10. The IPO, which is expected to take the company’s valuation to $20 billion, is poised to become the biggest IPO in the history of the Indian capital markets. The Indian government has been working on a cryptocurrency bill for quite some time. Initially, the government was considering a bill to ban cryptocurrencies, like bitcoin. However, recent reports suggest that the government is now planning to regulate the crypto sector. The crypto legislation will be “distinct and unique,” one lawmaker said. Last month, Finance Ministry officials reportedly said that crypto regulation would most likely come around by February. Meanwhile, the country’s central bank, the Reserve Bank of India (RBI), still has “serious concerns” about cryptocurrency, which have been communicated to the government. The RBI also said that a digital rupee model may be unveiled by the end of the year.''',
                   '''U.S. lawmakers have called on the Securities and Exchange Commission (SEC) to approve bitcoin spot exchange-traded funds (ETFs). Since the SEC has approved the trading of bitcoin futures ETFs, the lawmakers said it “should no longer have concerns with bitcoin spot ETFs and should show a similar willingness to permit the trading of bitcoin spot ETFs.” Lawmakers Urge SEC to Permit Trading of Bitcoin Spot ETFs. U.S. Representatives Tom Emmer and Darren Soto sent a bipartisan letter to the chairman of the Securities and Exchange Commission (SEC), Gary Gensler, Wednesday regarding bitcoin exchange-traded funds (ETFs). So far, the SEC has approved two bitcoin futures ETFs but has yet to approve any bitcoin spot ETF. Rep. Emmer said: The SEC’s approach to cryptocurrency regulation has been unacceptable. While the trading of bitcoin futures ETFs is a great step forward for the millions of American investors who have been demanding regulatory clarity, it does not make sense that bitcoin spot ETFs cannot also commence trading. Noting that the SEC approved two bitcoin futures ETFs, Reps Emmer and Soto wrote: “We question why, if you are comfortable allowing trading in an ETF based on derivatives contracts, you are not equally or more comfortable allowing trading to commence in ETFs based on spot bitcoin.” They explained, “Bitcoin spot ETFs are based directly on the asset, which inherently provides more protection for investors,” adding that futures products “are potentially much more volatile than a bitcoin spot ETF and may impose substantially higher fees on investors.” Referencing the SEC’s previous reasoning for disallowing spot bitcoin ETFs, the lawmakers asserted that “Since the SEC no longer has concerns with bitcoin futures ETFs,” then “it presumably has changed its view about the underlying spot bitcoin market because bitcoin futures are, by definition, a derivative of the underlying Bitcoin spot market.” They continued: The SEC should no longer have concerns with bitcoin spot ETFs and should show a similar willingness to permit the trading of bitcoin spot ETFs. The letter also notes that while the SEC continues to deny bitcoin ETFs, “numerous spot bitcoin investment vehicles have been offered,” with more than $40 billion in assets under management (AUM). “However, because these products have been unable to register as ETFs with the SEC, public trading typically occurs at a value that is not equivalent to net asset value, and in fact, these products have recently been trading at steep discounts to their net asset value,” the congressmen stressed. They elaborated: Permitting futures-based ETFs while simultaneously continuing to deny spot-based ETFs would further perpetuate these discounts and clearly go against the SEC’s core mission of protecting investors. The letter concludes: “The SEC is in a position to approve bitcoin futures ETFs, as reflected by the trading of these products, so it should also be in a position to approve bitcoin spot ETFs.”''',
                   '''Star Financial Bank (Star Bank) says it has become the first bank in the U.S. state of Indiana to offer bitcoin services to customers. The services will be offered through the New York Digital Investment Group and Alkami platform. Customers will have the ability to buy and sell bitcoin via the Star Mobile Banking App. ‘First Bank in the State of Indiana to Offer Bitcoin Trading Services’. Star Financial Bank (Star Bank) announced Tuesday: Star is excited to offer customers the ability to buy and sell bitcoin via the Star Mobile Banking App. “We’re launching this new offering as a closed beta,” the bank added. Star Bank is an Indiana-based community bank. Its parent company, Star Financial Group, has $2.80 billion in assets with 36 locations in central and northeast Indiana, according to its website. The bitcoin services are provided by the New York Digital Investment Group (NYDIG) via Alkami, a cloud-based digital banking solutions provider for U.S. banks and credit unions. NYDIG is the bitcoin investment arm of Stone Ridge Asset Management. According to the announcement published on Star Bank’s website: Star Bank is the first bank in the state of Indiana to offer bitcoin trading services to customers. The bank’s customers will have the option to “acquire, sell, hold, and manage bitcoin alongside their traditional assets,” the announcement details. Alkami’s founder and chief strategy and sales officer, Stephen Bohanon, explained that his company “helps financial institutions achieve digital banking success by delivering the most advanced cloud-based digital banking platform on the market.” He opined: “Early technology adopters appreciate the importance of embracing bitcoin opportunities.”''',
                   '''On November 1, at block height 707,639, a blockchain parser caught two bitcoin whale transfers that moved approximately 19,876 bitcoin worth $1.2 billion in the mix of 2,819 transactions. Interestingly, the owner used a similar splitting mechanism the old school mining whale blockchain parsers caught, spending strings of 20 block rewards throughout 2020 and 2021. Bitcoin Whale Watching. Bitcoin whales are mysterious animals because in a blockchain world of pseudonymity we only see them when they move. Last year and this year as well, Bitcoin.com News has hunted a specific whale entity that spent thousands of bitcoin mined in 2010. Every single time the whale spent the decade-old bitcoin that sat idle the whole time, the entity spent exactly 20 block rewards or 1,000 BTC. After the transfer, the wallets holding 1,000 BTC dispersed the funds into smaller-sized wallets. According to the creator of btcparser.com, the close to 20K BTC transferred at block height 707,639 on November 1 shared similar splitting mechanics with the “20×50 awakenings.” The blockchain parser’s owner would guess that the entity spending the two transactions could be the same person or organization. The special transactions stemming from block height 707,639 derived from the bitcoin addresses “15kEr” and “1PfaY.” The 15kEr address transferred 9,900.87 BTC, while 1PfaY spent 9,975.31 BTC. Spending a String of 20,000 BTC — 2 Bitcoin Whale Transactions Move Over $1.2 Billion. One of the bitcoin addresses’ splitting methods (pictured left) and the blockchain explorer data recording the 19,876 bitcoin moved on Monday (pictured right). The two transactions were filtered among 2,819 BTC transfers with 6,406 inputs recorded in block 707,639. The output total in that block was 9,587 with 78,704.53 BTC dispersed. The two transactions stemming from 15kEr and 1PfaY, represented more than 25% of the BTC processed in block 707,639. After the funds were sent, the nearly 20K BTC was split into 200 wallets with 100 BTC each. Then the bitcoin whale’s funds were split again into much smaller wallets until they finally consolidated into different amounts. The 2 Transactions Leveraged Moderate Privacy Tactics — 50 Bitcoin Block Reward From 2011 Spent 59 Blocks Later. Data from blockchair.com’s Privacy-o-meter for Bitcoin Transactions tool shows the wallet that sent the 9,975.31 BTC got a score of 60 or “moderate.” This is because matched addresses were identified and blockchair.com’s tool notes that “matching significantly reduces the anonymity of addresses.” The 9,900.87 BTC spend suffers from the same tracking vulnerabilities as matched addresses were also identified. Alongside the close to 20K BTC transfer in two separate transactions, 59 blocks later 50 sleeping bitcoins that had sat idle since April 28, 2011, were transferred at block height 707,698. The 50 BTC sat idle for over ten years since the day they were mined and when they were transferred, the exchange rate for the block reward of 50 BTC was just over $3 million. Blockchair.com’s privacy tool indicates the transaction got a score of 0 or “critical.” A critical score means that the tool “identified issues [that] significantly endanger the privacy of the parties involved.”''']

    # lemmatization allows to reduce word variance
    bitcoin_news = lemmatize_docs(bitcoin_news)
    keywords = keywords_extractor_sklearn(bitcoin_news, num_words = 10)
    print('KEYWORDS USED:', keywords, end='\n\n')
    topic_cluster(keywords, bitcoin_news)
    

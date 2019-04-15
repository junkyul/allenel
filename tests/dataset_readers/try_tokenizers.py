from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SpacyWordSplitter

spacy_splitter = SpacyWordSplitter()
"""
from https://en.wikipedia.org/?curid=993546

m.03xh50
993546
Japan_national_football_team
2
2
JAPAN
SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .

organization.sports_team

Oleg_Shatskiku Hiroshige_Yanagimoto Uzbek Igor_Shkvyrin Bitar Syria Nader_Jokhadar Asian_Cup China JAPAN Shu_Kamo Kuwait Syrians FIFA Salem_Bitar CHINA Japan Uzbekistan Takuya_Takagi Syrian AL-AIN Soviet Hassan_Abbas World_Cup Asian_Games United_Arab_Emirates UAE Chinese Nadim_Ladki South_Korea Indonesia

"""

test_sentence = """
SOCCER-JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT.	
"""

sp_toks = spacy_splitter.split_words(test_sentence)
print(sp_toks[0].text == "SOCCER")
print(sp_toks)
# [SOCCER, -, JAPAN, GET, LUCKY, WIN, ,, CHINA, IN, SURPRISE, DEFEAT, .]



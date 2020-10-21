import time
from collections import namedtuple

import pytest

from haystack.generator.transformers import RAGenerator, RAGeneratorType

QUESTIONS = [
    "who got the first nobel prize in physics",
    "when is the next deadpool movie being released",
    "which mode is used for short wave broadcast service",
    "who is the owner of reading football club",
    "when is the next scandal episode coming out",
    "when is the last time the philadelphia won the superbowl",
    "what is the most current adobe flash player version",
    "how many episodes are there in dragon ball z",
    "what is the first step in the evolution of the eye",
    "where is gall bladder situated in human body",
    "what is the main mineral in lithium batteries",
    "who is the president of usa right now",
    "where do the greasers live in the outsiders",
    "panda is a national animal of which country",
    "what is the name of manchester united stadium",
]

EXPECTED_SEQUENCE_OUTPUTS = [
    " albert einstein",
    " june 22, 2018",
    " amplitude modulation",
    " tim besley ( chairman )",
    " june 20, 2018",
    " 1980",
    " 7.0",
    " 8",
    " reticular formation",
    " walls of the abdomen",
    " spodumene",
    " obama",
    " grainger's compound",
    " japan",
    " old trafford stadium",
]

EXPECTED_TOKEN_OUTPUTS = [
    " albert einstein",
    " september 22, 2017",
    " amplitude modulation",
    " stefan persson",
    " april 20, 2018",
    " the 1970s",
    " 7.1. 2",
    " 13",
    " step by step",
    " stomach",
    " spodumene",
    " obama",
    " northern new jersey",
    " india",
    " , england",
    # TODO: From RAG Token generator out is below one
    # " united stadium",
]

DOC_DICT_LIST = [
    {'embedding': None,
     'id': '277bd64c-70aa-4a6b-9884-8f09b5f705fb',
     'meta': {'name': '"Albert Einstein"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'to Einstein in 1922. Footnotes Citations Albert Einstein Albert '
             'Einstein (; ; 14 March 1879 – 18 April 1955) was a German-born '
             'theoretical physicist who developed the theory of relativity, one of '
             'the two pillars of modern physics (alongside quantum mechanics). His '
             'work is also known for its influence on the philosophy of science. '
             'He is best known to the general public for his mass–energy '
             'equivalence formula , which has been dubbed "the world\'s most '
             'famous equation". He received the 1921 Nobel Prize in Physics "for '
             'his services to theoretical physics, and especially for his '
             'discovery of the law of'}
    ,
    {'embedding': None,
     'id': '60f23f03-e46a-4ee5-9c62-200e2a6c7f10',
     'meta': {'name': '"Albert Einstein"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Albert Einstein Albert Einstein (; ; 14 March 1879 – 18 April 1955) '
             'was a German-born theoretical physicist who developed the theory of '
             'relativity, one of the two pillars of modern physics (alongside '
             'quantum mechanics). His work is also known for its influence on the '
             'philosophy of science. He is best known to the general public for '
             'his mass–energy equivalence formula , which has been dubbed "the '
             'world\'s most famous equation". He received the 1921 Nobel Prize in '
             'Physics "for his services to theoretical physics, and especially for '
             'his discovery of the law of the photoelectric effect", a pivotal '
             'step'}
    ,
    {'embedding': None,
     'id': 'ff18029d-38bf-41c8-a8e6-80003ea03ae2',
     'meta': {'name': '"Albert Einstein"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'observations were published in the international media, making '
             'Einstein world-famous. On 7 November 1919, the leading British '
             'newspaper "The Times" printed a banner headline that read: '
             '"Revolution in Science – New Theory of the Universe – Newtonian '
             'Ideas Overthrown". In 1920, he became a Foreign Member of the Royal '
             'Netherlands Academy of Arts and Sciences. In 1922, he was awarded '
             'the 1921 Nobel Prize in Physics "for his services to Theoretical '
             'Physics, and especially for his discovery of the law of the '
             'photoelectric effect". While the general theory of relativity was '
             'still considered somewhat controversial, the citation also does not'}
    ,
    {'embedding': None,
     'id': '653f378f-f186-4571-9531-1202838451ec',
     'meta': {'name': '"Albert Einstein"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'model for depictions of mad scientists and absent-minded professors; '
             'his expressive face and distinctive hairstyle have been widely '
             'copied and exaggerated. "Time" magazine\'s Frederic Golden wrote '
             'that Einstein was "a cartoonist\'s dream come true". Many popular '
             'quotations are often misattributed to him. Einstein received '
             'numerous awards and honors and in 1922 he was awarded the 1921 Nobel '
             'Prize in Physics "for his services to Theoretical Physics, and '
             'especially for his discovery of the law of the photoelectric '
             'effect". None of the nominations in 1921 met the criteria set by '
             'Alfred Nobel, so the 1921 prize was carried forward and awarded'}
    ,
    {'embedding': None,
     'id': 'a944f5c0-c4bf-491b-8849-3e5c2c9d3a1a',
     'meta': {'name': '"Alfred Nobel"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'was adopted as the standard technology for mining in the "Age of '
             'Engineering" bringing Nobel a great amount of financial success, '
             'though at a significant cost to his health. An offshoot of this '
             "research resulted in Nobel's invention of ballistite, the precursor "
             'of many modern smokeless powder explosives and still used as a '
             "rocket propellant. In 1888 Alfred's brother Ludvig died while "
             'visiting Cannes and a French newspaper erroneously published '
             "Alfred's obituary. It condemned him for his invention of dynamite "
             'and is said to have brought about his decision to leave a better '
             'legacy after his death. The obituary stated,'}
    ,
    {'embedding': None,
     'id': '03f703ae-3021-4709-b29d-47d212fb4154',
     'meta': {'name': '"Akira Kurosawa"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'for 2020. Patrick Frater writing for "Variety" magazine in May 2017 '
             'stated that another two unfinished films by Kurosawa were planned, '
             'with "Silvering Spear" to start filming in 2018. In September 2011, '
             "it was reported that remake rights to most of Kurosawa's movies and "
             'unproduced screenplays were assigned by the Akira Kurosawa 100 '
             "Project to the L.A.-based company Splendent. Splendent's chief "
             'Sakiko Yamada, stated that he aimed to "help contemporary '
             'film-makers introduce a new generation of moviegoers to these '
             'unforgettable stories". Kurosawa Production Co., established in '
             "1959, continues to oversee many of the aspects of Kurosawa's legacy. "
             "The director's son,"}
    ,
    {'embedding': None,
     'id': '78fcd8d1-1309-464f-9c08-8b75fb3c9509',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a near bezel-less design along with wireless charging. On September '
             '12, 2018, Apple introduced the iPhone XS, iPhone XS Max and iPhone '
             'XR. The iPhone XS and iPhone XS Max features Super Retina displays, '
             'a faster and improved dual camera system that offers breakthrough '
             'photo and video features, the first 7-nanometer chip in a smartphone '
             '— the A12 Bionic chip with next-generation Neural Engine — faster '
             'Face ID, wider stereo sound and introduces Dual SIM to iPhone. The '
             'iPhone XR comes in an all-screen glass and aluminium design with the '
             'most advanced LCD in a smartphone featuring a 6.1-inch Liquid'}
    ,
    {'embedding': None,
     'id': '6bf72a73-7784-4468-818c-f33e68af7d93',
     'meta': {'name': '"Akira Kurosawa"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'through the Second World War and beyond. The narrative centers on '
             'yearly birthday celebrations with his former students, during which '
             'the protagonist declares his unwillingness to die just yet—a theme '
             "that was becoming increasingly relevant for the film's 81-year-old "
             'creator. Filming began in February 1992 and wrapped by the end of '
             'September. Its release on April 17, 1993, was greeted by an even '
             'more disappointed reaction than had been the case with his two '
             'preceding works. Kurosawa nevertheless continued to work. He wrote '
             'the original screenplays "The Sea is Watching" in 1993 and "After '
             'the Rain" in 1995. While putting'}
    ,
    {'embedding': None,
     'id': 'a77884a0-aa6d-42b6-add2-f734c6c35cd8',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '2016, Apple introduced the iPhone 7 and the iPhone 7 Plus, which '
             'feature improved system and graphics performance, add water '
             'resistance, a new rear dual-camera system on the 7 Plus model, and, '
             'controversially, remove the 3.5 mm headphone jack. On September 12, '
             '2017, Apple introduced the iPhone 8 and iPhone 8 Plus, standing as '
             'evolutionary updates to its previous phones with a faster processor, '
             'improved display technology, upgraded camera systems and wireless '
             'charging. The company also announced iPhone X, which radically '
             'changes the hardware of the iPhone lineup, removing the home button '
             'in favor of facial recognition technology and featuring'}
    ,
    {'embedding': None,
     'id': 'a15db189-993d-4fe3-a331-431aa842b9a8',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a faster processor, and brighter display. On September 12, 2017, '
             'Apple introduced the Apple Watch Series 3 featuring LTE cellular '
             'connectivity, giving the wearable independence from an iPhone except '
             'for the setup process. On September 12, 2018, Apple introduced the '
             'Apple Watch Series 4, featuring new display, electrocardiogram and '
             'fall detection. At the 2007 Macworld conference, Jobs demonstrated '
             'the Apple TV (previously known as the iTV), a set-top video device '
             'intended to bridge the sale of content from iTunes with '
             "high-definition televisions. The device links up to a user's TV and "
             'syncs, either via Wi-Fi or a wired network, with'}
    ,
    {'embedding': None,
     'id': 'fe75af12-805b-4b3c-9845-f1b593f54a4a',
     'meta': {'name': '"Amplitude modulation"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Amplitude modulation Amplitude modulation (AM) is a modulation '
             'technique used in electronic communication, most commonly for '
             'transmitting information via a radio carrier wave. In amplitude '
             'modulation, the amplitude (signal strength) of the carrier wave is '
             'varied in proportion to that of the message signal being '
             'transmitted. The message signal is, for example, a function of the '
             'sound to be reproduced by a loudspeaker, or the light intensity of '
             'pixels of a television screen. This technique contrasts with '
             'frequency modulation, in which the frequency of the carrier signal '
             'is varied, and phase modulation, in which its phase is varied. AM '
             'was'}
    ,
    {'embedding': None,
     'id': '2400eced-d6c6-4764-8993-47e3d8001f7e',
     'meta': {'name': '"Amplitude modulation"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'AM transmission (refer to Figure 2, but only considering positive '
             'frequencies) is twice the bandwidth of the modulating (or '
             '"baseband") signal, since the upper and lower sidebands around the '
             'carrier frequency each have a bandwidth as wide as the highest '
             'modulating frequency. Although the bandwidth of an AM signal is '
             'narrower than one using frequency modulation (FM), it is twice as '
             'wide as single-sideband techniques; it thus may be viewed as '
             'spectrally inefficient. Within a frequency band, only half as many '
             'transmissions (or "channels") can thus be accommodated. For this '
             'reason analog television employs a variant of single-sideband (known '
             'as'}
    ,
    {'embedding': None,
     'id': '39d1ae99-8d88-4fd4-85aa-993d5a0936d6',
     'meta': {'name': '"Amplitude modulation"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'the earliest modulation method used to transmit voice by radio. It '
             'was developed during the first quarter of the 20th century beginning '
             "with Landell de Moura and Reginald Fessenden's radiotelephone "
             'experiments in 1900. It remains in use today in many forms of '
             'communication; for example it is used in portable two-way radios, '
             'VHF aircraft radio, citizens band radio, and in computer modems in '
             'the form of QAM. "AM" is often used to refer to mediumwave AM radio '
             'broadcasting. In electronics and telecommunications, modulation '
             'means varying some aspect of a continuous wave carrier signal with '
             'an information-bearing modulation waveform, such as'}
    ,
    {'embedding': None,
     'id': '58a556fd-3c9d-43cf-a178-ede29569e5d6',
     'meta': {'name': '"Amplitude modulation"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'in which ones and zeros are represented by the presence or absence '
             'of a carrier. On-off keying is likewise used by radio amateurs to '
             'transmit Morse code where it is known as continuous wave (CW) '
             'operation, even though the transmission is not strictly '
             '"continuous." A more complex form of AM, quadrature amplitude '
             'modulation is now more commonly used with digital data, while making '
             'more efficient use of the available bandwidth. In 1982, the '
             'International Telecommunication Union (ITU) designated the types of '
             'amplitude modulation: Although AM was used in a few crude '
             'experiments in multiplex telegraph and telephone transmission in the'}
    ,
    {'embedding': None,
     'id': 'bb1b5c5c-9f5a-4fd9-8361-71d50984991b',
     'meta': {'name': '"Amplitude modulation"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'reduced-carrier transmission or DSB-RC) to use in the demodulation '
             'process. Even with the carrier totally eliminated in double-sideband '
             'suppressed-carrier transmission, carrier regeneration is possible '
             "using a Costas phase-locked loop. This doesn't work however for "
             'single-sideband suppressed-carrier transmission (SSB-SC), leading to '
             'the characteristic "Donald Duck" sound from such receivers when '
             'slightly detuned. Single sideband is nevertheless used widely in '
             'amateur radio and other voice communications both due to its power '
             'efficiency and bandwidth efficiency (cutting the RF bandwidth in '
             'half compared to standard AM). On the other hand, in medium wave and '
             'short wave broadcasting, standard AM with the full carrier'}
    ,
    {'embedding': None,
     'id': '0a555b8d-8d8c-408f-9f82-27ee7618d9c1',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'In England and Wales, the grounds used are: Old Trafford in '
             'Manchester (1884), The Oval in Kennington, South London (1884); '
             "Lord's in St John's Wood, North London (1884); Trent Bridge at West "
             'Bridgford, Nottinghamshire (1899), Headingley in Leeds (1899); '
             'Edgbaston in Birmingham (1902); Sophia Gardens in Cardiff, Wales '
             '(2009); and the Riverside Ground in Chester-le-Street, County Durham '
             '(2013). One Test was also held at Bramall Lane in Sheffield in 1902. '
             'Traditionally the final Test of the series is played at the Oval. '
             'Sophia Gardens and the Riverside were excluded as Test grounds '
             'between the years of 2020 and 2024 and'}
    ,
    {'embedding': None,
     'id': 'a148755d-ee75-4850-bbc5-e64f48d6eff8',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'captain Alastair Cook being bowled out for 0 (his first duck in 26 '
             'innings as captain), Australia looked to be in with a significant '
             'chance of a win, keeping its series hopes alive. By lunch England '
             'were 37–3, but on resumption of play only 3 balls were bowled before '
             'rain stopped play. This rain persisted and, at 16:40, the captains '
             'shook hands and the match was declared a draw. With England 2–0 up '
             'with two Tests to play, England retained the Ashes on 5 August 2013. '
             'In the Fourth Test, England won the toss and batted first, putting '
             'on 238'}
    ,
    {'embedding': None,
     'id': 'c065272a-d65e-4be4-b8b5-45c34f2d81b3',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'the quickest - in terms of balls faced - a team has been bowled out '
             'in the first innings of a Test match. With victory by an innings and '
             '78 runs on the morning of the third day of the Fourth Test, England '
             'regained the Ashes. During the buildup, the 2017–18 Ashes series was '
             'regarded as a turning point for both sides. Australia were '
             'criticised for being too reliant on captain Steve Smith and '
             'vice-captain David Warner, while England was said to have a shoddy '
             'middle to lower order. Off the field, England all-rounder Ben Stokes '
             'was ruled out of'}
    ,
    {'embedding': None,
     'id': '905d4229-1170-47e5-96d2-459411bc9d3f',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'The Ashes The Ashes is a Test cricket series played between England '
             'and Australia. The Ashes are regarded as being held by the team that '
             'most recently won the Test series. If the test series is drawn, the '
             'team that currently holds the Ashes retains the trophy. The term '
             'originated in a satirical obituary published in a British newspaper, '
             '"The Sporting Times", immediately after Australia\'s 1882 victory at '
             'The Oval, its first Test win on English soil. The obituary stated '
             'that English cricket had died, and "the body will be cremated and '
             'the ashes taken to Australia". The mythical ashes'}
    ,
    {'embedding': None,
     'id': '6c97c96e-1382-4455-a879-716c28c1c499',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'therefore will not host an Ashes Test until at least 2027. Trent '
             'Bridge is also not due to host an Ashes Test in 2019 or 2023. The '
             'popularity and reputation of the cricket series has led to other '
             'sports or games, and/or their followers, using the name "Ashes" for '
             'contests between England and Australia. The best-known and '
             'longest-running of these events is the rugby league rivalry between '
             'Great Britain and Australia (see rugby league "Ashes"). Use of the '
             'name "Ashes" was suggested by the Australian team when rugby league '
             'matches between the two countries commenced in 1908. Other examples '
             'included'}
    ,
    {'embedding': None,
     'id': 'b707cf75-6dc7-49fa-83bc-8540351592f0',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'an app that allows iPad and iPhone users to make and edit videos. '
             'The app provides a way to produce short videos to share with other '
             'users on the Messages app, Instagram, Facebook and other social '
             'networks. Apple also introduced Live Titles for Clips that allows '
             'users to add live animated captions and titles using their voice. In '
             'May 2017, Apple refreshed two of its website designs. Their public '
             'relations "Apple Press Info" website was changed to an "Apple '
             'Newsroom" site, featuring a greater emphasis on imagery and '
             'therefore lower information density, and combines press releases, '
             'news items, and photos.'}
    ,
    {'embedding': None,
     'id': '9f2eeb36-b7ce-4558-93c9-391f9ad8dc38',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '2018 after being delayed from its initial December 2017 release. It '
             'also features 7 tweeters in the base, a four-inch woofer in the top, '
             'and six microphones for voice control and acoustic optimization On '
             'September 12, 2018, Apple announced that HomePod is adding new '
             'features—search by lyrics, set multiple timers, make and receive '
             'phone calls, Find My iPhone, Siri Shortcuts—and Siri languages. '
             'Apple develops its own operating systems to run on its devices, '
             'including macOS for Mac personal computers, iOS for its iPhone, iPad '
             'and iPod Touch smartphones and tablets, watchOS for its Apple Watch '
             'smartwatches, and tvOS for its'}
    ,
    {'embedding': None,
     'id': 'b6fc7931-9878-4cff-a97a-0269a0381f68',
     'meta': {'name': '"A Modest Proposal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a letter to a local Aspen newspaper informing them that, on '
             'Christmas Eve, he was going to use napalm to burn a number of dogs '
             'and hopefully any humans they find. The letter protests against the '
             'burning of Vietnamese people occurring overseas. The 2012 film '
             '"Butcher Boys," written by Kim Henkel, is said to be loosely based '
             'on Jonathan Swift\'s "A Modest Proposal." The film\'s opening scene '
             'takes place in a restaurant named "J. Swift\'s". On November 30, '
             "2017, Jonathan Swift's 350th birthday, The Washington Post published "
             "a column entitled 'Why Alabamians should consider eating Democrats' "
             'babies", by the humorous'}
    ,
    {'embedding': None,
     'id': '22ad41d8-b60d-41ac-9303-3bcbcecf0597',
     'meta': {'name': '"Apollo 11"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '"Friendship 7". "Columbia" was moved in 2017 to the NASM Mary Baker '
             'Engen Restoration Hangar at the Steven F. Udvar-Hazy Center in '
             'Chantilly, Virginia, to be readied for a four-city tour titled '
             '"Destination Moon: The Apollo 11 Mission". This included Space '
             'Center Houston from October 14, 2017 to March 18, 2018, the Saint '
             'Louis Science Center from April 14 to September 3, 2018, the Senator '
             'John Heinz History Center in Pittsburgh from September 29, 2018 to '
             'February 18, 2019, and the Seattle Museum of Flight from March 16 to '
             "September 2, 2019. For 40 years Armstrong's and Aldrin's space suits"}
    ,
    {'embedding': None,
     'id': 'dfc01989-65a8-4712-aa1c-bd549e5e8582',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a faster processor, and brighter display. On September 12, 2017, '
             'Apple introduced the Apple Watch Series 3 featuring LTE cellular '
             'connectivity, giving the wearable independence from an iPhone except '
             'for the setup process. On September 12, 2018, Apple introduced the '
             'Apple Watch Series 4, featuring new display, electrocardiogram and '
             'fall detection. At the 2007 Macworld conference, Jobs demonstrated '
             'the Apple TV (previously known as the iTV), a set-top video device '
             'intended to bridge the sale of content from iTunes with '
             "high-definition televisions. The device links up to a user's TV and "
             'syncs, either via Wi-Fi or a wired network, with'}
    ,
    {'embedding': None,
     'id': '92c281c4-d587-4d77-8a53-55ae31a5b24e',
     'meta': {'name': '"American Revolutionary War"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '14, 1784. Copies were sent back to Europe for ratification by the '
             'other parties involved, the first reaching France in March 1784. '
             'British ratification occurred on April 9, 1784, and the ratified '
             'versions were exchanged in Paris on May 12, 1784. The war formally '
             'concluded on September 3, 1783. The last British troops departed New '
             'York City on November 25, 1783, marking the end of British rule in '
             'the new United States. The total loss of life throughout the '
             'conflict is largely unknown. As was typical in wars of the era, '
             'diseases such as smallpox claimed more lives than battle.'}
    ,
    {'embedding': None,
     'id': '009addd9-6479-47dd-8c53-6dd638629fae',
     'meta': {'name': '"American Civil War"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'ranging from the reenactment of battles, to statues and memorial '
             'halls erected, to films being produced, to stamps and coins with '
             'Civil War themes being issued, all of which helped to shape public '
             'memory. This varied advent occurred in greater proportions on the '
             "100th and 150th anniversary. Hollywood's take on the war has been "
             'especially influential in shaping public memory, as seen in such '
             'film classics as "Birth of a Nation" (1915), "Gone with the Wind" '
             '(1939), and more recently "Lincoln" (2012). Ken Burns produced a '
             'notable PBS series on television titled "The Civil War" (1990). It '
             'was digitally remastered'}
    ,
    {'embedding': None,
     'id': 'a4cab8c8-b9ce-4cf6-8f01-5acdf6fd4b4e',
     'meta': {'name': '"American Revolutionary War"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'signed the Treaty of Paris in which Great Britain agreed to '
             'recognize the sovereignty of the United States and formally end the '
             'war. French involvement had proven decisive, but France made few '
             'gains and incurred crippling debts. Spain made some territorial '
             'gains but failed in its primary aim of recovering Gibraltar. The '
             'Dutch were defeated on all counts and were compelled to cede '
             'territory to Great Britain. In India, the war against Mysore and its '
             'allies concluded in 1784 without any territorial changes. Parliament '
             'passed the Stamp Act in 1765. Colonists condemned the tax because '
             'their rights as Englishmen protected'}
    ,
    {'embedding': None,
     'id': 'ad09ddb7-5d06-4c6b-80d0-641051d6e0af',
     'meta': {'name': '"American Civil War"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'been achieved and that Reconstruction should end. They ran a '
             'presidential ticket in 1872 but were decisively defeated. In 1874, '
             'Democrats, primarily Southern, took control of Congress and opposed '
             'any more reconstruction. The Compromise of 1877 closed with a '
             'national consensus that the Civil War had finally ended. With the '
             'withdrawal of federal troops, however, whites retook control of '
             'every Southern legislature; the Jim Crow period of '
             'disenfranchisement and legal segregation was about to begin. The '
             'Civil War is one of the central events in American collective '
             'memory. There are innumerable statues, commemorations, books and '
             'archival collections. The memory includes'}
    ,
    {'embedding': None,
     'id': '377252fd-f605-4702-a0ab-0dd3a1604983',
     'meta': {'name': '"American Revolutionary War"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'desire to support the Americans the following year, hoping to weaken '
             "Britain's empire. The Portuguese threat was neutralized in the "
             'Spanish–Portuguese War (1776–77). On 12 April 1779, Spain signed the '
             'Treaty of Aranjuez with France and went to war against Britain. '
             'Spain sought to recover Gibraltar and Menorca in Europe, as well as '
             'Mobile and Pensacola in Florida, and also to expel the British from '
             'Central America. Meanwhile, George III had given up on subduing '
             'America while Britain had a European war to fight. He did not '
             'welcome war with France, but he believed that Britain had made all '
             'necessary'}
    ,
    {'embedding': None,
     'id': '582a60ae-5efb-47f0-98be-00b5f703677f',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Flash to store games, programs, and to cache the current media '
             'playing. The release also coincided with the opening of a separate '
             'Apple TV App Store and a new Siri Remote with a glass touchpad, '
             'gyroscope, and microphone. At the September 12, 2017 event, Apple '
             'released a new 4K Apple TV with the same form factor as the 4th '
             'Generation model. The 4K model is powered by the A10X SoC designed '
             'in-house that also powers their second-generation iPad Pro. The 4K '
             "model also has support for high dynamic range. Apple's first smart "
             'speaker, the HomePod was released on February 9,'}
    ,
    {'embedding': None,
     'id': '5149d8a7-dd4c-4cd8-9de4-bad1f7cdd7c3',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a near bezel-less design along with wireless charging. On September '
             '12, 2018, Apple introduced the iPhone XS, iPhone XS Max and iPhone '
             'XR. The iPhone XS and iPhone XS Max features Super Retina displays, '
             'a faster and improved dual camera system that offers breakthrough '
             'photo and video features, the first 7-nanometer chip in a smartphone '
             '— the A12 Bionic chip with next-generation Neural Engine — faster '
             'Face ID, wider stereo sound and introduces Dual SIM to iPhone. The '
             'iPhone XR comes in an all-screen glass and aluminium design with the '
             'most advanced LCD in a smartphone featuring a 6.1-inch Liquid'}
    ,
    {'embedding': None,
     'id': 'be97620b-96d7-4b69-8ab3-5e6263188377',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'include access to the iTunes Store to rent movies and TV shows '
             '(purchasing has been discontinued), streaming from internet video '
             'sources, including YouTube and Netflix, and media streaming from an '
             'iTunes library. Apple also reduced the price of the device to $99. A '
             'third generation of the device was introduced at an Apple event on '
             'March 7, 2012, with new features such as higher resolution (1080p) '
             'and a new user interface. At the September 9, 2015, event, Apple '
             'unveiled an overhauled Apple TV, which now runs a variant of macOS, '
             'called tvOS, and contains 32GB or 64 GB of NAND'}
    ,
    {'embedding': None,
     'id': '42c2362f-2808-4fab-a39e-05147174201e',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '2018 after being delayed from its initial December 2017 release. It '
             'also features 7 tweeters in the base, a four-inch woofer in the top, '
             'and six microphones for voice control and acoustic optimization On '
             'September 12, 2018, Apple announced that HomePod is adding new '
             'features—search by lyrics, set multiple timers, make and receive '
             'phone calls, Find My iPhone, Siri Shortcuts—and Siri languages. '
             'Apple develops its own operating systems to run on its devices, '
             'including macOS for Mac personal computers, iOS for its iPhone, iPad '
             'and iPod Touch smartphones and tablets, watchOS for its Apple Watch '
             'smartwatches, and tvOS for its'}
    ,
    {'embedding': None,
     'id': '3a6448e5-b4bc-4044-8d42-9b73026bdb4b',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'a faster processor, and brighter display. On September 12, 2017, '
             'Apple introduced the Apple Watch Series 3 featuring LTE cellular '
             'connectivity, giving the wearable independence from an iPhone except '
             'for the setup process. On September 12, 2018, Apple introduced the '
             'Apple Watch Series 4, featuring new display, electrocardiogram and '
             'fall detection. At the 2007 Macworld conference, Jobs demonstrated '
             'the Apple TV (previously known as the iTV), a set-top video device '
             'intended to bridge the sale of content from iTunes with '
             "high-definition televisions. The device links up to a user's TV and "
             'syncs, either via Wi-Fi or a wired network, with'}
    ,
    {'embedding': None,
     'id': '43b4b97b-c447-42d8-b205-c0c040151041',
     'meta': {'name': '"Apollo 8"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'as early as May 1967 that there would be at least four additional '
             'missions. Apollo 8 was planned as the "D" mission, a test of the LM '
             'in a low Earth orbit in December 1968 by James McDivitt, David '
             "Scott, and Russell Schweickart, while Borman's crew would fly the "
             '"E" mission, a more rigorous LM test in an elliptical medium Earth '
             'orbit as Apollo 9, in early 1969. The "F" Mission would test the CSM '
             'and LM in lunar orbit, and the "G" mission would be the finale, the '
             'Moon landing. Production of the LM fell behind schedule, and when'}
    ,
    {'embedding': None,
     'id': '8091634d-9c57-41ca-8bfe-5e473e477795',
     'meta': {'name': 'Anime'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Osamu Tezuka adapted and simplified many Disney animation techniques '
             'to reduce costs and to limit the number of frames in productions. He '
             'intended this as a temporary measure to allow him to produce '
             'material on a tight schedule with inexperienced animation staff. '
             '"Three Tales", aired in 1960, was the first anime shown on '
             'television. The first anime television series was "Otogi Manga '
             'Calendar", aired from 1961 to 1964. The 1970s saw a surge of growth '
             'in the popularity of "manga", Japanese comic books and graphic '
             'novels, many of which were later animated. The work of Osamu Tezuka '
             'drew particular attention:'}
    ,
    {'embedding': None,
     'id': '6ace4548-94ea-45da-8dd2-412a509e9372',
     'meta': {'name': '"Akira Kurosawa"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Akira Kurosawa Akira Kurosawa (, "Kurosawa Akira"; March 23, 1910 – '
             'September 6, 1998) was a Japanese film director and screenwriter, '
             'who directed 30 films in a career spanning 57 years. He is regarded '
             'as one of the most important and influential filmmakers in the '
             'history of cinema. Kurosawa entered the Japanese film industry in '
             '1936, following a brief stint as a painter. After years of working '
             'on numerous films as an assistant director and scriptwriter, he made '
             'his debut as a director during World War II with the popular action '
             'film "Sanshiro Sugata" (a.k.a. "Judo Saga"). After the war,'}
    ,
    {'embedding': None,
     'id': '9268d3db-374a-4ef5-9928-6ba50d227c5f',
     'meta': {'name': '"Aquarius (constellation)"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Aquarii among its 12 stars. 88, 89, and 98 Aquarii represent '
             '"Fou-youe", the axes used as weapons and for hostage executions. '
             'Also in Aquarius is "Loui-pi-tchin", the ramparts that stretch from '
             '29 and 27 Piscium and 33 and 30 Aquarii through Phi, Lambda, Sigma, '
             'and Iota Aquarii to Delta, Gamma, Kappa, and Epsilon Capricorni. '
             'Near the border with Cetus, the axe "Fuyue" was represented by three '
             'stars; its position is disputed and may have instead been located in '
             'Sculptor. "Tienliecheng" also has a disputed position; the 13-star '
             'castle replete with ramparts may have possessed Nu and Xi Aquarii '
             'but may'}
    ,
    {'embedding': None,
     'id': 'a135c68d-2735-4043-a41b-aab0a2d72122',
     'meta': {'name': 'Anime'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'mind, but are also aimed at the general music market, and therefore '
             'often allude only vaguely or not at all to the themes or plot of the '
             'series. Pop and rock songs are also sometimes used as incidental '
             'music ("insert songs") in an episode, often to highlight '
             'particularly important scenes. The animation industry consists of '
             'more than 430 production companies with some of the major studios '
             'including Toei Animation, Gainax, Madhouse, Gonzo, Sunrise, Bones, '
             'TMS Entertainment, Nippon Animation, P.A.Works, Studio Pierrot and '
             'Studio Ghibli. Many of the studios are organized into a trade '
             'association, The Association of Japanese Animations. There'}
    ,
    {'embedding': None,
     'id': 'bb27639c-2de1-4915-ae49-fee5326a618d',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'seen with the naked eye, and also includes superficial anatomy or '
             'surface anatomy, the study by sight of the external body features. '
             'Microscopic anatomy is the study of structures on a microscopic '
             'scale, along with histology (the study of tissues), and embryology '
             '(the study of an organism in its immature condition). Anatomy can be '
             'studied using both invasive and non-invasive methods with the goal '
             'of obtaining information about the structure and organization of '
             'organs and systems. Methods used include dissection, in which a body '
             'is opened and its organs studied, and endoscopy, in which a video '
             'camera-equipped instrument is inserted'}
    ,
    {'embedding': None,
     'id': 'a7801f71-0fb8-4dab-9081-815b3c381af1',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'related disciplines, and they are often studied together. Human '
             'anatomy is one of the essential basic sciences that are applied in '
             'medicine. The discipline of anatomy is divided into macroscopic and '
             'microscopic anatomy. Macroscopic anatomy, or gross anatomy, is the '
             "examination of an animal's body parts using unaided eyesight. Gross "
             'anatomy also includes the branch of superficial anatomy. Microscopic '
             'anatomy involves the use of optical instruments in the study of the '
             'tissues of various structures, known as histology, and also in the '
             'study of cells. The history of anatomy is characterized by a '
             'progressive understanding of the functions of the'}
    ,
    {'embedding': None,
     'id': 'f880d71e-1bf0-4ff7-8f2c-221f32e44d01',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Magnetic resonance imaging, computed tomography, and ultrasound '
             'imaging have all enabled examination of internal structures in '
             'unprecedented detail to a degree far beyond the imagination of '
             'earlier generations. Anatomy Anatomy (Greek anatomē, "dissection") '
             'is the branch of biology concerned with the study of the structure '
             'of organisms and their parts. Anatomy is a branch of natural science '
             'which deals with the structural organization of living things. It is '
             'an old science, having its beginnings in prehistoric times. Anatomy '
             'is inherently tied to developmental biology, embryology, comparative '
             'anatomy, evolutionary biology, and phylogeny, as these are the '
             'processes by which anatomy is'}
    ,
    {'embedding': None,
     'id': '6667b5e2-06ad-48b5-9a57-1f7f0a9a1230',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'tail which continues the spinal cord and vertebrae but not the gut. '
             'The mouth is found at the anterior end of the animal, and the anus '
             'at the base of the tail. The defining characteristic of a vertebrate '
             'is the vertebral column, formed in the development of the segmented '
             'series of vertebrae. In most vertebrates the notochord becomes the '
             'nucleus pulposus of the intervertebral discs. However, a few '
             'vertebrates, such as the sturgeon and the coelacanth retain the '
             'notochord into adulthood. Jawed vertebrates are typified by paired '
             'appendages, fins or legs, which may be secondarily lost. The limbs '
             'of vertebrates'}
    ,
    {'embedding': None,
     'id': 'a696c90a-18de-4b09-a5c3-f56069432cea',
     'meta': {'name': '"Augustin-Jean Fresnel"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'theory and stellar aberration. He was informed that he was trying to '
             'break down open doors (""il enfonçait des portes ouvertes""), and '
             'directed to classical works on optics. On 12 July 1815, as Fresnel '
             'was about to leave Paris, Arago left him a note on a new topic: '
             'Fresnel would not have ready access to these works outside Paris, '
             'and could not read English. But, in Mathieu — with a point-source of '
             'light made by focusing sunlight with a drop of honey, a crude '
             'micrometer of his own construction, and supporting apparatus made by '
             'a local locksmith — he began'}
    ,
    {'embedding': None,
     'id': 'd782a56b-c5d9-4e0b-b276-a89eddfda0e8',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'to venom glands as most spiders are venomous. They have a second '
             'pair of appendages called pedipalps attached to the cephalothorax. '
             'These have similar segmentation to the legs and function as taste '
             'and smell organs. At the end of each male pedipalp is a spoon-shaped '
             'cymbium that acts to support the copulatory organ. In 1600 BCE, the '
             'Edwin Smith Papyrus, an Ancient Egyptian medical text, described the '
             'heart, its vessels, liver, spleen, kidneys, hypothalamus, uterus and '
             'bladder, and showed the blood vessels diverging from the heart. The '
             'Ebers Papyrus (c. 1550 BCE) features a "treatise on the heart", with '
             'vessels'}
    ,
    {'embedding': None,
     'id': 'd1b27544-b5b9-465b-a12d-99124b1851fe',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'in such organs as sea anemone tentacles and the body wall of sea '
             'cucumbers. Skeletal muscle contracts rapidly but has a limited range '
             'of extension. It is found in the movement of appendages and jaws. '
             'Obliquely striated muscle is intermediate between the other two. The '
             'filaments are staggered and this is the type of muscle found in '
             'earthworms that can extend slowly or make rapid contractions. In '
             'higher animals striated muscles occur in bundles attached to bone to '
             'provide movement and are often arranged in antagonistic sets. Smooth '
             'muscle is found in the walls of the uterus, bladder, intestines, '
             'stomach,'}
    ,
    {'embedding': None,
     'id': '2f2d7743-43ce-4ab8-879d-7c7cb110d8d3',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'organs, including the stomach. All vertebrates have a similar basic '
             'body plan and at some point in their lives, mostly in the embryonic '
             'stage, share the major chordate characteristics; a stiffening rod, '
             'the notochord; a dorsal hollow tube of nervous material, the neural '
             'tube; pharyngeal arches; and a tail posterior to the anus. The '
             'spinal cord is protected by the vertebral column and is above the '
             'notochord and the gastrointestinal tract is below it. Nervous tissue '
             'is derived from the ectoderm, connective tissues are derived from '
             'mesoderm, and gut is derived from the endoderm. At the posterior end '
             'is a'}
    ,
    {'embedding': None,
     'id': '61feca69-a96d-415b-83b1-3f3af7b45ae9',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'pouch where it latches on to a nipple and completes its development. '
             'Humans have the overall body plan of a mammal. Humans have a head, '
             'neck, trunk (which includes the thorax and abdomen), two arms and '
             'hands, and two legs and feet. Generally, students of certain '
             'biological sciences, paramedics, prosthetists and orthotists, '
             'physiotherapists, occupational therapists, nurses, podiatrists, and '
             'medical students learn gross anatomy and microscopic anatomy from '
             'anatomical models, skeletons, textbooks, diagrams, photographs, '
             'lectures and tutorials, and in addition, medical students generally '
             'also learn gross anatomy through practical experience of dissection '
             'and inspection of cadavers. The study of microscopic anatomy'}
    ,
    {'embedding': None,
     'id': '2dc325bb-b5ca-42f8-9a80-fdc2ebe8683e',
     'meta': {'name': 'Anatomy'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'has three pairs of segmented legs, one pair each for the three '
             'segments that compose the thorax and one or two pairs of wings. The '
             'abdomen is composed of eleven segments, some of which may be fused '
             'and houses the digestive, respiratory, excretory and reproductive '
             'systems. There is considerable variation between species and many '
             'adaptations to the body parts, especially wings, legs, antennae and '
             'mouthparts. Spiders a class of arachnids have four pairs of legs; a '
             'body of two segments—a cephalothorax and an abdomen. Spiders have no '
             'wings and no antennae. They have mouthparts called chelicerae which '
             'are often connected'}
    ,
    {'embedding': None,
     'id': 'b33bf038-99e0-4a7c-97f2-f931f032d94b',
     'meta': {'name': '"Alkali metal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'million (ppm) or 25 micromolar. Its diagonal relationship with '
             'magnesium often allows it to replace magnesium in ferromagnesium '
             'minerals, where its crustal concentration is about 18 ppm, '
             'comparable to that of gallium and niobium. Commercially, the most '
             'important lithium mineral is spodumene, which occurs in large '
             'deposits worldwide. Rubidium is approximately as abundant as zinc '
             'and more abundant than copper. It occurs naturally in the minerals '
             'leucite, pollucite, carnallite, zinnwaldite, and lepidolite, '
             'although none of these contain only rubidium and no other alkali '
             'metals. Caesium is more abundant than some commonly known elements, '
             'such as antimony, cadmium, tin, and tungsten,'}
    ,
    {'embedding': None,
     'id': 'f7a6a707-7c22-4458-b37d-60b28232a202',
     'meta': {'name': '"Alkali metal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Lithium salts have to be extracted from the water of mineral '
             'springs, brine pools, and brine deposits. The metal is produced '
             'electrolytically from a mixture of fused lithium chloride and '
             'potassium chloride. Sodium occurs mostly in seawater and dried '
             'seabed, but is now produced through electrolysis of sodium chloride '
             'by lowering the melting point of the substance to below 700 °C '
             'through the use of a Downs cell. Extremely pure sodium can be '
             'produced through the thermal decomposition of sodium azide. '
             'Potassium occurs in many minerals, such as sylvite (potassium '
             'chloride). Previously, potassium was generally made from the '
             'electrolysis of'}
    ,
    {'embedding': None,
     'id': '8576a773-1f60-462b-8e61-f15dbbd3c364',
     'meta': {'name': '"Alkali metal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': "and the Dead Sea. Despite their near-equal abundance in Earth's "
             'crust, sodium is far more common than potassium in the ocean, both '
             "because potassium's larger size makes its salts less soluble, and "
             'because potassium is bound by silicates in soil and what potassium '
             'leaches is absorbed far more readily by plant life than sodium. '
             'Despite its chemical similarity, lithium typically does not occur '
             'together with sodium or potassium due to its smaller size. Due to '
             'its relatively low reactivity, it can be found in seawater in large '
             'amounts; it is estimated that seawater is approximately 0.14 to 0.25 '
             'parts per'}
    ,
    {'embedding': None,
     'id': 'b46d5483-83b2-4a94-a720-2ac2571c7f9f',
     'meta': {'name': '"Alkali metal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'of lithium in humans has yet to be identified. Sodium and potassium '
             'occur in all known biological systems, generally functioning as '
             'electrolytes inside and outside cells. Sodium is an essential '
             'nutrient that regulates blood volume, blood pressure, osmotic '
             'equilibrium and pH; the minimum physiological requirement for sodium '
             'is 500 milligrams per day. Sodium chloride (also known as common '
             'salt) is the principal source of sodium in the diet, and is used as '
             'seasoning and preservative, such as for pickling and jerky; most of '
             'it comes from processed foods. The Dietary Reference Intake for '
             'sodium is 1.5 grams per day, but'}
    ,
    {'embedding': None,
     'id': 'a535c855-048f-4b3e-8b7e-97e98aad7ad5',
     'meta': {'name': '"Alkali metal"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'react with carbon dioxide to form the alkali metal carbonate and '
             'oxygen gas, which allows them to be used in submarine air purifiers; '
             'the presence of water vapour, naturally present in breath, makes the '
             'removal of carbon dioxide by potassium superoxide even more '
             'efficient. All the stable alkali metals except lithium can form red '
             'ozonides (MO) through low-temperature reaction of the powdered '
             'anhydrous hydroxide with ozone: the ozonides may be then extracted '
             'using liquid ammonia. They slowly decompose at standard conditions '
             'to the superoxides and oxygen, and hydrolyse immediately to the '
             'hydroxides when in contact with water. Potassium, rubidium, and'}
    ,
    {'embedding': None,
     'id': '2ad39f14-c64c-4f4e-a052-b1f28023c374',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'CEO. Two months later, Jobs died, marking the end of an era for the '
             'company. Apple is well known for its size and revenues. Its '
             'worldwide annual revenue totaled $265billion for the 2018 fiscal '
             "year. Apple is the world's largest information technology company by "
             "revenue and the world's third-largest mobile phone manufacturer "
             'after Samsung and Huawei. In August 2018, Apple became the first '
             'public U.S. company to be valued at over US$1 trillion. The company '
             'employs 123,000 full-time employees and maintains 504 retail stores '
             "in 24 countries . It operates the iTunes Store, which is the world's "
             'largest music retailer.'}
    ,
    {'embedding': None,
     'id': '67edb401-7bc0-4147-852c-6dd1d5234ab4',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'of $250 billion, officially confirmed by Apple as specifically '
             '$256.8 billion a few days later. , Apple is the largest publicly '
             'traded corporation in the world by market capitalization. On August '
             '2, 2018, Apple became the first publicly traded U.S. company to '
             'reach a $1 trillion market value. Apple is currently ranked #4 on '
             'the Fortune 500 rankings of the largest United States corporations '
             'by total revenue. Apple has created subsidiaries in low-tax places '
             'such as Ireland, the Netherlands, Luxembourg and the British Virgin '
             'Islands to cut the taxes it pays around the world. According to "The '
             'New York Times,"'}
    ,
    {'embedding': None,
     'id': '4628c5c2-da83-4569-a776-ea53dfed172c',
     'meta': {'name': '"America the Beautiful"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'broadcast of the Tiangong-1 launch. The song is often included in '
             'songbooks in a wide variety of religious congregations in the United '
             'States. Bing Crosby included the song in a medley on his album "101 '
             'Gang Songs" (1961). In 1976, while the United States celebrated its '
             'bicentennial, a soulful version popularized by Ray Charles peaked at '
             'number 98 on the US R&B Charts. Ray Charles did this again in 1984 '
             'to re-elect Ronald Reagan. Ray Charles did this yet again in Miami, '
             'Florida in 1999. Three different renditions of the song have entered '
             'the Hot Country Songs charts. The first'}
    ,
    {'embedding': None,
     'id': '0ecd6bf7-cfd8-47a8-9224-085f22b7e057',
     'meta': {'name': '"Apple Inc."'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'A mid-October 2013 announcement revealed that Burberry executive '
             'Angela Ahrendts will commence as a senior vice president at Apple in '
             "mid-2014. Ahrendts oversaw Burberry's digital strategy for almost "
             'eight years and, during her tenure, sales increased to about US$3.2 '
             'billion and shares gained more than threefold. Alongside Google '
             'vice-president Vint Cerf and AT&T CEO Randall Stephenson, Cook '
             'attended a closed-door summit held by President Obama on August 8, '
             '2013, in regard to government surveillance and the Internet in the '
             'wake of the Edward Snowden NSA incident. On February 4, 2014, Cook '
             'met with Abdullah Gül, the President of Turkey, in'}
    ,
    {'embedding': None,
     'id': '74827729-26aa-4322-9dcd-b5f2924ae9e4',
     'meta': {'name': '"American (word)"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '"the United States of North America" in the first sentence, then '
             '"the said United States" afterwards; "the United States of America" '
             'and "the United States of North America" derive from "the United '
             'Colonies of America" and "the United Colonies of North America". The '
             'Treaty of Peace and Amity of September 5, 1795, between the United '
             'States and the Barbary States contains the usages "the United States '
             'of North America", "citizens of the United States", and "American '
             'Citizens". U.S. President George Washington, in his 1796 "Farewell '
             'Address", declaimed that "The name of American, which belongs to you '
             'in your national capacity,'}
    ,
    {'embedding': None,
     'id': '39d0a2ab-967a-4a40-8695-e2bc6d133b55',
     'meta': {'name': '"Achill Island"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'for outdoor adventure activities, like surfing, kite-surfing and sea '
             'kayaking. Fishing and watersports are also popular. Sailing regattas '
             'featuring a local vessel type, the Achill Yawl, have been popular '
             'since the 19th century, though most present-day yawls, unlike their '
             'traditional working boat ancestors, have been structurally modified '
             "to promote greater speed under sail. The island's waters and "
             'underwater sites are occasionally visited by scuba divers, though '
             "Achill's unpredictable weather generally has precluded a "
             'commercially successful recreational diving industry. In 2011, the '
             "population was 2,569. The island's population has declined from "
             'around 6,000 before the Great Hunger. The table below reports'}
    ,
    {'embedding': None,
     'id': 'acec8ef0-46a2-43c4-8eb8-2be5dc890bef',
     'meta': {'name': '"Achill Island"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'these houses since the time of the Famine, however, the families '
             'that moved to Dooagh and their descendants, continued to use the '
             "village as a 'booley village'. This means that during the summer "
             'season, the younger members of the family, teenage boys and girls, '
             'would take the cattle to graze on the hillside and they would stay '
             'in the houses of the Deserted Village. This custom continued until '
             'the 1940s. Boolying was also carried out in other areas of Achill, '
             'including Annagh on Croaghaun mountain and in Curraun. At Ailt, '
             'Kildownet, you can see the remains of a similar deserted'}
    ,
    {'embedding': None,
     'id': 'a7238eaa-539c-4c2c-92bc-88515f906894',
     'meta': {'name': '"Alexander Graham Bell"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'status as: "I am not one of those hyphenated Americans who claim '
             'allegiance to two countries." Despite this declaration, Bell has '
             'been proudly claimed as a "native son" by all three countries he '
             'resided in: the United States, Canada, and the United Kingdom. By '
             '1885, a new summer retreat was contemplated. That summer, the Bells '
             'had a vacation on Cape Breton Island in Nova Scotia, spending time '
             'at the small village of Baddeck. Returning in 1886, Bell started '
             'building an estate on a point across from Baddeck, overlooking Bras '
             'd\'Or Lake. By 1889, a large house, christened "The Lodge" was'}
    ,
    {'embedding': None,
     'id': 'dda0a838-fc86-44e7-8c8c-704c96795602',
     'meta': {'name': 'Alaska'},
     'probability': None,
     'question': None,
     'score': None,
     'text': "Jack London's novel and starring Ethan Hawke, was filmed in and "
             'around Haines. Steven Seagal\'s 1994 "On Deadly Ground", starring '
             'Michael Caine, was filmed in part at the Worthington Glacier near '
             'Valdez. The 1999 John Sayles film "Limbo", starring David '
             'Strathairn, Mary Elizabeth Mastrantonio, and Kris Kristofferson, was '
             'filmed in Juneau. The psychological thriller "Insomnia", starring Al '
             'Pacino and Robin Williams, was shot in Canada, but was set in '
             'Alaska. The 2007 film directed by Sean Penn, "Into The Wild", was '
             'partially filmed and set in Alaska. The film, which is based on the '
             'novel of the same name, follows'}
    ,
    {'embedding': None,
     'id': '88b91118-3224-4651-9bba-dce534932773',
     'meta': {'name': '"Ainu people"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'D1b is found throughout the Japanese Archipelago, but with very high '
             'frequencies among the Ainu of Hokkaido in the far north, and to a '
             'lesser extent among the Ryukyuans in the Ryukyu Islands of the far '
             'south. The only places outside Japan in which Y-haplogroup D is '
             'common are Tibet in China and the Andaman Islands in the Indian '
             'Ocean. A study by Tajima "et al." (2004) found two out of a sample '
             'of sixteen (or 12.5%) Ainu men to belong to Haplogroup C-M217, which '
             'is the most common Y-chromosome haplogroup among the indigenous '
             'populations of Siberia and Mongolia. Hammer'}
    ,
    {'embedding': None,
     'id': '08bcea43-9881-40c9-a19a-717f7ab6091a',
     'meta': {'name': 'Ashoka'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '("Ashoka Chakra") from its base was placed onto the center of the '
             'National Flag of India. The capital contains four lions (Indian / '
             'Asiatic Lions), standing back to back, mounted on a short '
             'cylindrical abacus, with a frieze carrying sculptures in high relief '
             'of an elephant, a galloping horse, a bull, and a lion, separated by '
             'intervening spoked chariot-wheels over a bell-shaped lotus. Carved '
             'out of a single block of polished sandstone, the capital was '
             "believed to be crowned by a 'Wheel of Dharma' (Dharmachakra "
             'popularly known in India as the "Ashoka Chakra"). The Sarnath pillar '
             'bears one of the'}
    ,
    {'embedding': None,
     'id': '23820a37-8cd9-44f5-b848-68d4e24e8bf4',
     'meta': {'name': 'Ashoka'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'pre-independence versions of the flag. The Ashoka Chakra can also '
             'been seen on the base of the Lion Capital of Ashoka which has been '
             'adopted as the National Emblem of India. The Ashoka Chakra was '
             'created by Ashoka during his reign. Chakra is a Sanskrit word which '
             'also means "cycle" or "self-repeating process". The process it '
             'signifies is the cycle of time—as in how the world changes with '
             'time. A few days before India became independent in August 1947, the '
             'specially-formed Constituent Assembly decided that the flag of India '
             'must be acceptable to all parties and communities. A flag with'}
    ,
    {'embedding': None,
     'id': '03fc95a1-dbfc-454a-8b57-779b2615da67',
     'meta': {'name': 'Ashoka'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Wheel of Dharma). The wheel has 24 spokes which represent the 12 '
             'Laws of Dependent Origination and the 12 Laws of Dependent '
             'Termination. The Ashoka Chakra has been widely inscribed on many '
             'relics of the Mauryan Emperor, most prominent among which is the '
             'Lion Capital of Sarnath and The Ashoka Pillar. The most visible use '
             'of the Ashoka Chakra today is at the centre of the National flag of '
             'the Republic of India (adopted on 22 July 1947), where it is '
             'rendered in a Navy-blue color on a White background, by replacing '
             'the symbol of Charkha (Spinning wheel) of the'}
    ,
    {'embedding': None,
     'id': '19bce840-1cdf-4cbd-a135-f3e18cf51099',
     'meta': {'name': 'Azerbaijan'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'The national animal of Azerbaijan is the Karabakh horse, a '
             'mountain-steppe racing and riding horse endemic to Azerbaijan. The '
             'Karabakh horse has a reputation for its good temper, speed, elegance '
             'and intelligence. It is one of the oldest breeds, with ancestry '
             'dating to the ancient world. However, today the horse is an '
             "endangered species. Azerbaijan's flora consists of more than 4,500 "
             'species of higher plants. Due to the unique climate in Azerbaijan, '
             'the flora is much richer in the number of species than the flora of '
             'the other republics of the South Caucasus. About 67 percent of the '
             'species growing'}
    ,
    {'embedding': None,
     'id': '5e1d2ba5-fde4-4aa9-9162-3917fad6dd22',
     'meta': {'name': '"Ainu people"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'both names. On June 6, 2008, the Japanese Diet passed a bipartisan, '
             'non-binding resolution calling upon the government to recognize the '
             'Ainu people as indigenous to Japan, and urging an end to '
             'discrimination against the group. The resolution recognised the Ainu '
             'people as "an indigenous people with a distinct language, religion '
             'and culture". The government immediately followed with a statement '
             'acknowledging its recognition, stating, "The government would like '
             'to solemnly accept the historical fact that many Ainu were '
             'discriminated against and forced into poverty with the advancement '
             'of modernization, despite being legally equal to (Japanese) people." '
             'As a result of'}
    ,
    {'embedding': None,
     'id': 'aa767c99-a3e1-4ff4-83d7-176959ac4b20',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'In England and Wales, the grounds used are: Old Trafford in '
             'Manchester (1884), The Oval in Kennington, South London (1884); '
             "Lord's in St John's Wood, North London (1884); Trent Bridge at West "
             'Bridgford, Nottinghamshire (1899), Headingley in Leeds (1899); '
             'Edgbaston in Birmingham (1902); Sophia Gardens in Cardiff, Wales '
             '(2009); and the Riverside Ground in Chester-le-Street, County Durham '
             '(2013). One Test was also held at Bramall Lane in Sheffield in 1902. '
             'Traditionally the final Test of the series is played at the Oval. '
             'Sophia Gardens and the Riverside were excluded as Test grounds '
             'between the years of 2020 and 2024 and'}
    ,
    {'embedding': None,
     'id': '139de2c1-eb15-4339-b692-2efc7d4b3bb4',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': '1978–79; 1981; 1985; 1989; 1993 and 1997). Australians have made 264 '
             'centuries in Ashes Tests, of which 23 have been scores over 200, '
             'while Englishmen have scored 212 centuries, of which 10 have been '
             'over 200. Australians have taken 10 wickets in a match on 41 '
             'occasions, Englishmen 38 times. The series alternates between the '
             'United Kingdom and Australia, and within each country each of the '
             'usually five matches is held at different grounds. In Australia, the '
             'grounds currently used are the Gabba in Brisbane (first staged an '
             'England–Australia Test in the 1932–33 season), Adelaide Oval '
             '(1884–85), the Melbourne Cricket'}
    ,
    {'embedding': None,
     'id': '1e97a20c-89de-4998-bd4a-10fb43a46c1e',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'Ground (MCG) (1876–77), and the Sydney Cricket Ground (SCG) '
             '(1881–82). A single Test was held at the Brisbane Exhibition Ground '
             'in 1928–29. Traditionally, Melbourne hosts the Boxing Day Test and '
             'Sydney hosts the New Year Test. Additionally the WACA in Perth '
             '(1970–71) hosted its final Ashes Test in 2017–18 and is due to be '
             'replaced by Perth Stadium for the 2021–22 series. Cricket Australia '
             'proposed that the 2010–11 series consist of six Tests, with the '
             'additional game to be played at Bellerive Oval in Hobart. The '
             'England and Wales Cricket Board declined and the series was played '
             'over five Tests.'}
    ,
    {'embedding': None,
     'id': '936cf65c-1cb2-4368-8a97-1ced4310c3cf',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'therefore will not host an Ashes Test until at least 2027. Trent '
             'Bridge is also not due to host an Ashes Test in 2019 or 2023. The '
             'popularity and reputation of the cricket series has led to other '
             'sports or games, and/or their followers, using the name "Ashes" for '
             'contests between England and Australia. The best-known and '
             'longest-running of these events is the rugby league rivalry between '
             'Great Britain and Australia (see rugby league "Ashes"). Use of the '
             'name "Ashes" was suggested by the Australian team when rugby league '
             'matches between the two countries commenced in 1908. Other examples '
             'included'}
    ,
    {'embedding': None,
     'id': '8700da49-a1f8-4470-92c0-ecfb817398c4',
     'meta': {'name': '"The Ashes"'},
     'probability': None,
     'question': None,
     'score': None,
     'text': 'The Ashes The Ashes is a Test cricket series played between England '
             'and Australia. The Ashes are regarded as being held by the team that '
             'most recently won the Test series. If the test series is drawn, the '
             'team that currently holds the Ashes retains the trophy. The term '
             'originated in a satirical obituary published in a British newspaper, '
             '"The Sporting Times", immediately after Australia\'s 1882 victory at '
             'The Oval, its first Test win on English soil. The obituary stated '
             'that English cricket had died, and "the body will be cremated and '
             'the ashes taken to Australia". The mythical ashes'}
    ,
]

DOCUMENTS = []
for doc_dict in DOC_DICT_LIST:
    DOCUMENTS.append(namedtuple('Document', ['id', 'text', 'meta', 'embedding', 'score', 'probability', 'question']))


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
def test_rag_token_generator(document_store, retriever):
    document_store.delete_all_documents()
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever)
    time.sleep(2)

    generator = RAGenerator(retriever=retriever, generator_type=RAGeneratorType.TOKEN)

    for idx, question in enumerate(QUESTIONS):
        retrieved_docs = retriever.retrieve(query=question, top_k=5)
        generated_docs = generator.predict(question=question, documents=retrieved_docs, top_k=1)
        answers = generated_docs["answers"]
        assert len(answers) == 1
        assert answers[0]["answer"] == EXPECTED_TOKEN_OUTPUTS[idx]

# IMPORTS
import numpy as np
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.optim as optim

# DEFINE DATA
class UserData:
    def __init__(self, attribute, angle = None):
        self.attribute = attribute
        self.angle = angle

    def get_image_data(self):
        # Images
        img_arx_desc = ["A small crowd walking by a church. All groups are walking in the same direction except one individual.",
                        "A group of people walking in front of a church during a tour.",
                        "Walking near the church",
                        "Handful of people passing by a medieval church, seen from the top of other building, street with car in the background",
                        "People walking in groups, all in the same direction but a guy walking on the opposite direction.",
                        "Group of people walking by the street in a public enclosure similar to a church.",
                        "Maybe some people are going to an event all together or maybe they are leaving an event. Maybe there is a tour going on.",
                        "A rooftop shot of a small crowd of mostly young people walking past a church with an open door",
                        "25 students walk and socialize in the courtyard of a church located in a city.",
                        "Approximately twenty people walk in pairs or in groups down a pedestrian zone in one direction.",
                        "a group of people walking in an old church's/castle's outdoor environment ,mostly in just one direction except only one person that is going the opposite direction .they have entered this environment moments before",
                        "A group of people walking into building complex",
                        "A small group of 20 young people with their professors walk through the church's yard going the same direction"]
        img_arx_si = ["Some are walking alone while most of the people seem to be part of different groups.",
                    "some are walking in a small group and talking to each other, some are walking alone some in threes",
                    "friendly",
                    "Regular",
                    "People sometimes talking to each other",
                    "Most of them are interacting with the people around, but there are a few persons walking alone.",
                    "They know each other. Probably they belong to the same family or social group.",
                    "Maybe they are all in the same group. Some of them have formed teams, other walk alone. But it is difficult to say from a static image. There is a group of people which seems to be the core of the group.",
                    "looks some friends and a family talking together and walking in a same direction (except one person ;) ).",
                    "They form 3 groups that walk and socialize. It looks like a school trip.",
                    "People seem to be engaging in conversation while walking down the street",
                    "cooperative",
                    "Some people walk together, but mostly isolated or small groups",
                    "They walk in the same direction, seems to be related, a large group that has partially dispersed"]
        img_uni_desc=["People walking and socializing in a square",
                    "A birds eye view of around 25 people casually walking in different directions in a park cement pathway in groups or alone. A small cantina under the shadow of a tree in the upper middle are of the pictures",
                    "people walking on a pedestrian area of a city",
                    "This is an image of people walking in a park, on the pavement. The crowd is not dance, people walk in different directions, they walk alone or form groups of maximum 5 people.",
                    "People are randomly walking through different directions in pairs and alone.",
                    "People walk and interact amongst themselves in park",
                    "A few people walking in a street on different directions. The camera looks at the people from above",
                    "Sidewalk with direct sunlight and two shaded areas. Some people are walking in groups, some are sitting.",
                    "in a park corner, people walk in all directions alone or in small groups."]
        img_uni_si=["Strong social interaction",
                    "Cooperation",
                    "small groups of people",
                    "People form groups, they keep some sort of social distance.",
                    "There is friendly interaction between people",
                    "People engage in small talk",
                    "some of them are together in small groups chatting.",
                    "Close interaction in small groups",
                    "casual"]
        img_eth_desc=["Bird view of a street in a developed city or town during winter, people walking casually.",
                    "11 people walk to the left, 6 people walk to the right, and 2 people stop on a street. There are two trees and one street light in the middle of the street.",
                    "City walkway next to tram line, with a few trees and benches, view from above, a few people walking alone or in pairs in the same direction",
                    "top down view of a wide pedestrian street with some tress, a bench, a two-lamp light-post, and several people walking",
                    "nothing",
                    "Realistic look-down view of a street next to the train with a few poeple walking on it",
                    "A top view image of people walking in opposite directions the platform of a metro station.",
                    "Some people walking in a street in different directions. These people can be walking in groups of two or alone.",
                    "Top view perspective of an outside and uncovered waiting area of a train/metro station. The picture shows 2 trees that have no leaves, so probably it is winter time, a bench and a lamp post. Lastly, there are approximately 15 people shown walking in various directions.",
                    "A drone or camera is looking at a street from a top-down view. People are walking in both directions individually, some people look like they are going to work."]
        img_eth_si=["strangers",
                    "cold",
                    "Not much social interaction, the people walking in pairs are talking to each other",
                    "most are walking in couples",
                    "none",
                    "Walking avoiding collisions.",
                    "Individuals and some pairs",
                    "The people are minding their own businesses",
                    "Difficult to say. If any social interaction is taking place then it is to a minimum.",
                    "People are not having any social interaction.",
                    "No interaction, people are walking in different directions or are standing."]
        img_desc=[img_arx_desc,img_uni_desc,img_eth_desc]
        img_source = ["arx","uni","eth"]
        my_dict = {"arx": img_arx_desc, "uni": img_uni_desc, "eth": img_eth_desc}

        if self.attribute == "descriptions":
            return my_dict
        elif self.attribute == "soicialInteractions":
            si_dict = {"arx": img_arx_si, "uni": img_uni_si, "eth": img_eth_si}
            return si_dict
        else:
            print("Error. No attribute: ",self.attribute," found.")

    def get_set_data(self):
        # Set of images
        sets_uni_desc=["A medium crowd of people gathering in groups, talking or walking in the street, next to a park.",
                    "people are gathering",
                    "Colored CCTV camera feed with people walking along an intersection at a university",
                    "The people on the street are walking on different directions",
                    "People passing on a broad walkway",
                    "People walking.",
                    "Aerial photo of people walking by the street placed in a public garden with grass and few trees.",
                    "People are walking on a pedestrian zone maybe inside a park. They are moving on both directions.",
                    "A crowded park where people walk, group and interact.",
                    "People are gathered in a square walking in one direction, then joined by increasingly more people.",
                    "a number of people in different groups walking in different direction in a park like environement",
                    "People walking in a city park. I'm not sure if something interesting happens, first and third images have an area of people grouped together but I can't see any detail.",
                    "a large group of small independent groups of people walking in the courtyard/gardens of a building"]
        sets_uni_si=["They seem to bee different groups of friends or colleagues hanging out together.",
                    "walking in different directions and gathering to a point",
                    "neutral",
                    "Regular",
                    "Small subsets of people sometimes interacting",
                    "Just walking with their partners/friends/groups",
                    "Groups of people: families, friends.",
                    "Some walk in groups or two, others in groups with more people, and some of them walk alone.",
                    "There is a crowd on the pavement of a park, walking in all directions, but most of the people are moving to the left.",
                    "People here form large and numerous groups. Few of them are walking alone.",
                    "I do not see any social interaction taking place, it seems that the pedestrians follow a specific route separately.",
                    "exchange",
                    "Random people, all seem walking on their own.",
                    "Independent, very small groups"]
        sets_eth_desc=["People walking on a sidewalk",
                    "A drone shot of 5-6 people walking on a pavement next to a tram station. The tram rails are in the upper part of the photos and the tram arriving",
                    "people walk on the sidewalk and a tram passes by",
                    "People are walking in the street, they empty the square.",
                    "People are walking on a pavement, with not interaction and moving from one picture to the other you less people appear",
                    "People wait for a train/tram as it approaches. The train/tram doors open and some people get on the train as others get off.",
                    "10 people in a street that move in diferent directions until they disappear from view. The camera is looking at the people from above.",
                    "People are walking on the sidewalk and a tram passes by.",
                    "People waiting and walking at the side of a train track"]
        sets_eth_si=["Minimal",
                    "non-existent",
                    "There are two groups of two people but beside from this it does not seem that there is any social interactions between people.",
                    "No interaction",
                    "People are strangers with each other.",
                    "no interaction at all",
                    "No social interaction",
                    "none"]
        sets_arx_desc=["view of pavement next to a church or a historic building, its cloudy or late afternoon, and people walk with a purpose (maybe leaving a mass or a meeting)",
                    "A group of people enter a place that has a historical building.",
                    "Sequence of images showing a courtyard around a church, with a small crowd of people gradually walking from one side to the other",
                    "a group of teenagers waling in one direction in the yard of an old christian orthodox church",
                    "people walking",
                    "Group of people walking next to an antique building. Generate three times the same image, with three diffeten displacements.",
                    "People walking near a church of medeival architecture style.",
                    "Three images of a small crowd walking through a passage next to a medieval building. These three pictures should represent the flow of the people walking from north-right to down-left",
                    "A small group of people moving towards a direction.",
                    "A group of people is walking together towards a place. It looks like they are touring in group.",
                    "People are moving outside a large building, probably a church. They walk at the same direction."]
        sets_arx_si=["acquaintances",
                    "friendly",
                    "The people in the crowd are all talking to each other",
                    "walking together and talkin",
                    "none",
                    "Sub-groups interactions",
                    "Tourists, walking in groups",
                    "The people seem to be part of a same group",
                    "It looks like a school trip or a guided tour guide of a specific site. There are some people leading the group followed by a bigger group of people followed by a small group of people.",
                    "They are probably touring in group, so they know each other. It looks like there are 3 sub groups.",
                    "Some people are in groups, but they are not interacting a lot."]
        
        sets_desc=[sets_uni_desc, sets_eth_desc, sets_arx_desc]
        sets_si=[sets_uni_si, sets_eth_si, sets_arx_si]
        sets_source = ["uni","eth","arx"]
        my_dict = {"uni": sets_uni_desc, "eth": sets_eth_desc, "arx": sets_arx_desc}

        if self.attribute == "descriptions":
            return my_dict
        elif self.attribute == "soicialInteractions":
            si_dict = {"uni": sets_uni_si, "eth": sets_eth_si, "arx": sets_arx_si}
            return si_dict
        else:
            print("Error. No attribute: ",self.attribute," found.")

    def get_video_data(self):
        # Real videos
        eth_1=["There is people passing by what seems to be a tramway station. Some are getting on and off the wagon, while others are waiting.",
                    "People are walking in different directions, some are passing by, some others are stopping, looking and moving along.",
                    "People getting on and off a train and others walking along a train station platform",
                    "People are walking on the street and some of them are getting into or our from the bus",
                    "People walking on a broad street, while some people are getting to or off a tram, seen from above.",
                    "People coming in and out from the public transport. Others just passing by.",
                    "Zenital video of a street with a tram stopped and some people getting on and off the tram",
                    "It is a public area where a tram just stopped. Some people take the tram or get off the tram. In front of the tram stop, there is a pavement where people are walking on both directions.",
                    "A crowd is walking on the wide pavement of a street In an autumn weather , people are walking along the sidewalk in both directions while some are getting on or off an electric bus or train stopped on the street.",
                    "A busy day on a train station that people hurry",
                    "People walk up and down a pedestrian zone while others get on and off a bus/tram or wait in line to get in. Some people walk in pairs talking, while others walk alone.",
                    "people walking in different group some entering a train and some just walking in bilinear direction on the sidewalk next to the train station",
                    "People getting of a tramway on a city street",
                    "Many people walk bidirectional, in a tram station in Europe, while some people are getting on and off the train"]
        eth_2=["People waiting and walking down a tramway or train station. A train arrives and some people get off it while other get in.",
                    "People are passing by walking in different directions, a tram is passing and stopping, people get in the tram and some others are getting off",
                    "People walking on a sidewalk",
                    "Some people are walking and some are waiting the train to get inside",
                    "People walking in a street and waiting for a tram on a tram stop. Eventually some people get in/out of a tram that arrived.",
                    "Same that video 1.3.1 but longer.",
                    "Zenital video of people walking by the street alone where a tram is stopped while some people gets on and off the tram.",
                    "People are moving in a pavement on both directions. Some people are waiting for the tram to arrive. When the tram arrives, they take the tram. At the same time some others are walking towards the pavement when as soon they get off the tram.",
                    "A top shot from a pavement including passing people in both direction and waiting people for city train in an autumn weather.",
                    "When the train arrives, some get on, get off, and some other are still waiting.",
                    "A chill day on a train station with a few people",
                    "passers-by walk up and down a pedestrian zone alone, one person is standing reading a newspaper, some people wait for a bus/tram to arrive. A tram arrives and people get off, about 10 people get on the tram and after a few seconds the tram departs. There are a couple more people waiting in line again for the next vehicle to arrive while passers-by keep walking up and down the zone.",
                    "most pedesterians use just one side of the road to walk no matter their direction and the side closer to the train station is being used mostly by the people getting off and on the train",
                    "A tramway stops on a city street, opens the door and some people get on.",
                    "Some individuals and small groups of people walk bidirectional near a tram station. Few people are standing waiting for the tram to arrive. As soon as the tram arrives, they get on the tram, while some other are getting of the tram and move to different directions"]
        zara_a=["People walking on a street corner opposite directions. Some people are going in and out of what seems to be a clothes store and there is one individual walking in the perpendicular direction.",
                "People are moving in different directions. Cars are also appearing moving across the street",
                "People walking in on a side walk in front of a fashion store",
                "People are walking in the street side",
                "People walking on a sidewalk, some crossing the street. Corner of a house and a couple of cars are visible.",
                "People walking with their partners/friends/groups.",
                "Groups of people walking by the street with shadows, cars parked on the road and shop windows.",
                "There is a clothes store which is located in a street's corner and in front of the store and on the right there is a pedestrian area where people are moving on all directions.",
                "A car starts moving after a while on the street and people are walking on the pavement in a moderate weather. They are walking in front of a corner building with vitrine and an alley crossing the street. Almost all of them are moving in the direction and to the right of the pedestrian.",
                "A street of shops where people walk and see the showcases.",
                "People walk up and down a pedestrian zone, alone or in pairs conversing, some people enter a shop located at the corner and cars are passing by.",
                "a car has blocked a percentage of the pedestrian walk so people are forced to slightly alter their course and after they pass the parked car they try to go back to their previous course ,this causes the pedestrians walking in opposite direction to face each other",
                "People walking on the sidewalk and a stopped car on the street starts moving",
                "Few small groups of people walk bidirectional on the pedestrian side near a main street, in the market of a modern city"]
        arx_1=["Group of people walking on a church yard",
            "High school students walking in a greek orhodox church yard",
            "A group of people walk together in the same direction",
            "Groups of students are walking into a historical monument's yard in the same direction.",
            "People are walking in groups talking to each other",
            "A group of people, mostly of young age, enter the courtyard of a large old stone building, while talking loudly amongst themselves.",
            "A group of 25 people gets in the backyard of a church",
            "A group of people are talking and walking together in the same direction.",
            "a group of people entering a patio from a street, talking and casually walking together."]
        arx_2=["Group of people walking and socializing on a church yard",
            "High School students walking on the outside of a greek orthodox church.",
            "A group of people walk together in the same direction",
            "A school group is walking in a historical monument' yard.",
            "People are walking in groups talking to each other",
            "Groups of people, primarily in their 20s, walk in the courtyard of an old stone building, while taking amongst them.",
            "a group of 25 people go across the backyard of a church",
            "The bigger group gets split into three and we see some more relevant interaction in the last group.",
            "A group of people walking though a building patio, chatting and walking in groups."]
        zara_b=["People walking on a sidewalk and enter in a store",
                "People walking on a pavement outside a clothes shop carrying shopping bags. Two cars are parked on the side of the shop and one in front of the pavement",
                "groups of 2 people croos each other as they walk in a commercial district",
                "Poeple are walking in a shopping district of a city, some of them are stopping in fonr tof windows to look at clothes in the shops",
                "people are walking in groups of two and in single units not interacting",
                "Some people walk on the sidewalk next to a corner clothing store, occasionally entering or leaving the store.",
                "people walking in pairs in a street in opposite directions",
                "People are walking in pairs along a sidewalk in both directions.",
                "People walking in the street by a shop window, with bags. Some show interest in the shop's contents."]
        uni_1=["Bird view of a city small square / junction of major streets. Its sunny/good weather, and there are a lot of people walking in all directions",
            "A crowd of college students were sitting, walking, standing and talking with people around on a wide street.",
            "Broad walkway in a park, view from above, a lot of people are walking in all directions, small stationary groups of people talking to each other",
            "top down video of many groups of people of 2 to 6 people walking, talking and sitting, in a wide pedestrian street",
            "people just walking around at a uni campus",
            "Generate video of a realistic look-down view of a street with people walking on it in different diretions and with sub-groups interactions.",
            "A video of tourists walking in a busy market street corner.",
            "A video of street videocamera at a corner of a square, where some people are talking while sit down, talking while stand up, or walking. The video footage is taken during a sunny day.",
            "Angled top view video of a crowd in a large sidewalk. people walking at a moderate speed in various directions alone or in various sized groups(no more than 6 people). There is also a small number of people alone or in various sized groups in static positions(sitting or standing)",
            "A top-down camera is looking at what looks to be a university campus. Small to medium size groups of people are talking, some of them walking. Groups do not walk in the same direction.",
            "Different people are standing or walking on a square. Some are talking with each other."]
        uni_2=["Bird view of a city small square / junction of major streets. Its sunny/good weather, and there are a lot of people walking in all directions. Its a bit weird that they are walking in chunks",
            "A crowd of college students were sitting, walking on a wide street. A small group of people were standing and talking in the middle of the street.",
            "Broad walkway in a park, view from above, a lot of people are walking in all directions",
            "top down video of people walking in all directions, from solo pedestrian to larger groups",
            "people waling in a more geometric like fashion",
            "Generate video of a realistic look-down view of a street with people walking on it mainly from rigth to left diretion and with sub-groups interactions.",
            "People are walking in a busy market street corner in the afternoon.",
            "A video of street videocamera at a corner of a square, where some people are talking while sit down, talking while stand up, or walking. The video footage is taken during a sunny day and a bit windy.",
            "Angled top view video of a crowd in a large sidewalk in a public space next to a patch of grass. People walking at a moderate pace in various directions alone or in various sized groups(no more than 6 people). There is also a small number of people alone or in various sized groups in static positions(sitting or standing)",
            "A top-down camera is looking at what looks to be a university campus. There is a medium size group walking to the left; most people walk to the left too. Other people walk in other directions.",
            "Many people are standing or walking on a square. Many walk in different directions."]
        zara_c=["Bird view of city or town people during noon in a sunny day, people walking casually, there is shadow from trees and parked cars",
                "A sparse group of people casually walking on a street that has one cloth store and two cars parking.",
                "City pedestrian pathway bordering a street intersection, view from above, a few cars parked alongside the pathway, a small number of people walking on the pathway in either direction",
                "Few people walking on a pedestrian street, some hold shopping bags",
                "people walking in a city",
                "Generate video of a realistic look-down view of a street with people walking on it in bi-diretions and with pairs interactions.",
                "People walking near parked cars in a commercial street footpath in a city in the afternoon.",
                "A video of street videocamera at a corner of a street with some cars parked, where some people walking in different directions. The video footage is taken during a sunny day.",
                "Angled top view video of people walking at a slow pace on a sidewalk outside a clothing store.",
                "It looks like a normal street, some people are walking in one direction, others leave from a shop and walk in the opposite direction.",
                "People are waling on the pavement by a street with shops, they pass by, some stop."]

        arx=[arx_1,arx_2]
        eth=[eth_1,eth_2]
        uni=[uni_1,uni_2]
        zara=[zara_a,zara_b,zara_c]
        # my_dict={"arx": arx_1+arx_2, "eth": eth_1+eth_2, "uni": uni_1+uni_2, "zara": zara_a+zara_b+zara_c}
        my_dict={"arx": arx, "eth": eth, "uni": uni, "zara": zara}

        return my_dict

    def get_synth_data(self):
        # Synthetic videos
        s_211_arx_c1=["A virtual crowd of agents, all walking in the same direction. Some go in groups while others walk alone.",
            "People are moving in the same direction",
            "People walking up and down an indoor pedestrian street.",
            "Some people are walking",
            "A medium-sized crowd of people walk in a computer-generated environment through a wide passage in one way. Some people walk faster, some are running or getting through the crowd.",
            "People walking in the same direction.",
            "Groups of people walking by the street in the same direction.",
            "People are trying to form a line and move towards one direction.",
            "a top angle view of a group of 20 to 30 people walking on a wide sidewalk in one direction with different speed.",
            "50 people are walking in a queue. Some of them form groups",
            "A group of approximately 20 people is heading down a route in one direction. The groups is divided in two parts, the one in front consisting of more people about 13-14 and the rest of the group follow behind.",
            "the Magarity of the people are walking in a big non-homogenous group except a small portion of them which start their way not following the same direction as the other people but changing their way and becoming a part of the main group at the end of the video",
            "About twenty people walking down a corridor, all in the same direction",
            "A bunch of people walk the same direction, some of them as individuals, while others as small groups."]
        s_212_arx_t1=["An aereal view of a virtual crowd of agents. They are all walking in the same direction inside a big corridor.",
                "people are moving in the same direction",
                "People walking up and down an indoor pedestrian street.",
                "Some people show to walking in different directions",
                "A handful of people walking one-way walking through a computer-generated passage, seen from the above.",
                "People walking. Most on the same direction, others look lost.",
                "Zenital video of people walking alone by the street mainly in the same direction.",
                "A group of people are moving in one direction.",
                "A top view of walking people with different speed to reach each other or a specific group and then they are walking in a specific and the same direction as other groups.",
                "3 groups of people explore a wide area",
                "People approximately 30 of them are walking in one direction down a route while a couple of them appear to be walking the opposite direction until they meet the rest of the group and then they turn around and follow the rest. People appear to be walking in clusters of around 5 or 10 people and some of them in pairs",
                "in this video most people are going towards the same direction, a small number of them walk in opposite direction if they do not interact with another person, the moment they interact with someone they change their way and follow the same path as everyone else",
                "A top-down view of about twenty people walking a corridor from left to right. One person seems to realize he/she took the wrong direction and turns around.",
                "A bunch of people walk the same direction, some of them as individuals, while others as small groups. They seem to be in a hurry"]
        s_211_arx_c2=["Group of people walking in the same direction",
                "A bird eye view of around 20 Virtual humans walking in different directions",
                "A group of people walk together in the same direction",
                "A group of virtual avatars are walking toward the same direction in a virtual environment and leave the scene. There are man and woman. Some of the avatars go to odd directions at the beginning and stand out from the crowd.",
                "walking in group, no interaction",
                "Some people walk in groups, on the side of a wall.",
                "A group of 20 synthetic people walking in a simplified and synthetic street in the same direction",
                "People are walking in the same direction, without seeming to interact with one another.",
                "a large group of people walking in the same direction, interacting among themselves."]
        s_222_zara_t=["People walking in various directions",
                "Top view of virtual humans walking in different directions",
                "groups of at most 2 people cross each other while walking in a square",
                "This is a top view video of virtual characters walking in various directions in a neutral basic white virtual environment. Some of the characters form social groups.",
                "People walking in units, no interaction",
                "Top view of people walking mostly in a bidirectional manner. Some people walk in place or wander around.",
                "Synthetic people walking through a synthetic plane from left to right and right to left, some walk alone and some walk in small groups at different velocities. The camera is looking down at the people",
                "People walking in different directions, sometimes in pairs",
                "Cenital view of some people walking in opposite directions, in pairs or alone."]
        s_212_arx_t2=["Bird view of grid-like floor with black lines that appear to be walking as they have other lines that mimic leg movements",
                "A crowd of people were walking on the street. Most of people were walking in the same direction and forming two big groups after a while.",
                "Plain walkway, view from above, with a small number of people walking mostly in the same direction, left to right",
                "top down video of people walking in one direction, some forming groups",
                "Look-down motion video in 3D render of far away looking people walking at different speeds from left to right",
                "An aerial video of people walking in the same direction in a large indoor office space.",
                "A top-view 14 seconds video of a rectangular grid space where around 20 people are walking from left to right. A few of these people are walking at a random direction at the start of the video, but correct their direction a few seconds later to follow the crowd.",
                "Top view perspective of a virtual environment with a simulated crowd. The video only contains human figures and no other elements. The crowd is mainly moving at a moderate pace from left to right with a few exceptions of figures that seem to want to walk towards a different direction but as soon as they come close to other figures the change their direction.",
                "A group of people is walking to the right except for some individual people, who are trying to walk to the left and eventually join the crowd and walk to the right.",
                "Some people are ice skating or birds are flying. Some collide."]
        s_221_zara_c=["Bird view of grid-like floor with artificially made people walking",
                "Several pairs of people walking to one direction. Several pairs of people and some single ones walking to the oppsite direction later.",
                "Plain walkway, showing a small number of people mostly in pairs or alone, walking either top to bottom, or bottom to top",
                "people in couples walking in both directions",
                "30 degrees look-down motion video in 3D render of middle distance looking people walking in pairs at the same speeds from top to down and down to top.",
                "A near-isometric view of people walking in opposite directions in a large indoor office space.",
                "A side-view 24 seconds video of a rectangular grid space where people are walking from left to right or viceversa. Almost all of them walk in groups of two and the rest walk alone, where each group walks at different velocities.",
                "Angled top view perspective of a virtual environment with a simulated figures. The video only contains human figures and no other elements. The human figures move in pairs or alone at a varying speeds from moderate to slightly fast pace, from the lower part of the video frame towards the higher part as well as from the higher part of the video frame towards the lower part.",
                "It looks like a normal street where people walk in both directions. One person looks like having problems walking.",
                "Few people are walking, some are walking faster than others."]
        
        corner = [s_211_arx_c1, s_211_arx_c2 , s_221_zara_c]
        top = [s_212_arx_t1, s_222_zara_t, s_212_arx_t2]
        arx = [s_211_arx_c1, s_212_arx_t1, s_211_arx_c2, s_212_arx_t2]
        zara = [s_222_zara_t, s_221_zara_c]
        my_dict = {"arx": arx,
                    "zara": zara}
        

        if self.angle == None:
            return my_dict
        else:
            angle_dict = {"corner": corner, "top:": top}
            return my_dict, angle_dict

    def tokenize_sentences(self, list_sentences, max_seq_len):
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        inputs = torch.zeros(len(list_sentences),max_seq_len).long()
        i = 0
        cutoff = 0
        for sentence in list_sentences:
            input_ids = tokenizer(sentence.strip('""'), return_tensors="pt").input_ids
            # print(sentence)
            # print(tokenizer.decode(input_ids[0]))
            # if i >= 5:
            #     exit()
            if input_ids.shape[1] <= max_seq_len:
                # discard response:
                inputs[i,0:input_ids.shape[1]] = input_ids[0] # [13]
                i += 1
            else:
                cutoff += 1
                # i += 1
        if cutoff != 0:
            inputs = inputs[:(len(list_sentences)-cutoff),:] 
            # print("WARNING: We have a cuttof of: ", cutoff) #TODO

        return inputs

    def get_img_si(self):
        img_arx_si = ["Some are walking alone while most of the people seem to be part of different groups.",
                    "some are walking in a small group and talking to each other, some are walking alone some in threes",
                    "friendly",
                    "Regular",
                    "People sometimes talking to each other",
                    "Most of them are interacting with the people around, but there are a few persons walking alone.",
                    "They know each other. Probably they belong to the same family or social group.",
                    "Maybe they are all in the same group. Some of them have formed teams, other walk alone. But it is difficult to say from a static image. There is a group of people which seems to be the core of the group.",
                    "looks some friends and a family talking together and walking in a same direction (except one person ;) ).",
                    "They form 3 groups that walk and socialize. It looks like a school trip.",
                    "People seem to be engaging in conversation while walking down the street",
                    "cooperative",
                    "Some people walk together, but mostly isolated or small groups",
                    "They walk in the same direction, seems to be related, a large group that has partially dispersed"]
        img_uni_si=["Strong social interaction",
                    "Cooperation",
                    "small groups of people",
                    "People form groups, they keep some sort of social distance.",
                    "There is friendly interaction between people",
                    "People engage in small talk",
                    "some of them are together in small groups chatting.",
                    "Close interaction in small groups",
                    "casual"]
        img_eth_si=["strangers",
                    "cold",
                    "Not much social interaction, the people walking in pairs are talking to each other",
                    "most are walking in couples",
                    "none",
                    "Walking avoiding collisions.",
                    "Individuals and some pairs",
                    "The people are minding their own businesses",
                    "Difficult to say. If any social interaction is taking place then it is to a minimum.",
                    "People are not having any social interaction.",
                    "No interaction, people are walking in different directions or are standing."]
        img_si=[img_arx_si,img_uni_si,img_eth_si]
        return img_si

    def get_set_si(self):
        sets_arx_si=["acquaintances",
                    "friendly",
                    "The people in the crowd are all talking to each other",
                    "walking together and talkin",
                    "none",
                    "Sub-groups interactions",
                    "Tourists, walking in groups",
                    "The people seem to be part of a same group",
                    "It looks like a school trip or a guided tour guide of a specific site. There are some people leading the group followed by a bigger group of people followed by a small group of people.",
                    "They are probably touring in group, so they know each other. It looks like there are 3 sub groups.",
                    "Some people are in groups, but they are not interacting a lot."]
        sets_uni_si=["They seem to bee different groups of friends or colleagues hanging out together.",
                    "walking in different directions and gathering to a point",
                    "neutral",
                    "Regular",
                    "Small subsets of people sometimes interacting",
                    "Just walking with their partners/friends/groups",
                    "Groups of people: families, friends.",
                    "Some walk in groups or two, others in groups with more people, and some of them walk alone.",
                    "There is a crowd on the pavement of a park, walking in all directions, but most of the people are moving to the left.",
                    "People here form large and numerous groups. Few of them are walking alone.",
                    "I do not see any social interaction taking place, it seems that the pedestrians follow a specific route separately.",
                    "exchange",
                    "Random people, all seem walking on their own.",
                    "Independent, very small groups"]
        sets_eth_si=["Minimal",
                    "non-existent",
                    "There are two groups of two people but beside from this it does not seem that there is any social interactions between people.",
                    "No interaction",
                    "People are strangers with each other.",
                    "no interaction at all",
                    "No social interaction",
                    "none"]
        sets_si = [sets_arx_si, sets_uni_si, sets_eth_si]
        return sets_si

    def get_all_videos(self):
        vid_dict = self.get_video_data()
        arx_video_0 = vid_dict["arx"][0]
        arx_video_1 = vid_dict["arx"][1] 
        uni_video_0 = vid_dict["uni"][0]
        uni_video_1 = vid_dict["uni"][1] 
        eth_video_0 = vid_dict["eth"][0]
        eth_video_1 = vid_dict["eth"][1]
        zara_video_1 = vid_dict["zara"][0]  
        zara_video_2 = vid_dict["zara"][1]  
        zara_video_3 = vid_dict["zara"][2]  
        video_sentences = arx_video_0 + arx_video_1 + uni_video_0 + uni_video_1 + eth_video_0 + eth_video_1 + zara_video_1 + zara_video_2 + zara_video_3
        return video_sentences

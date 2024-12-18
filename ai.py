from gitcg import (
    Deck, Player, Game, CreateParam, 
    ActionRequest, ActionResponse, 
    ChooseActiveRequest, ChooseActiveResponse, 
    RerollDiceRequest, RerollDiceResponse, 
    SelectCardRequest, SelectCardResponse, 
    SwitchHandsRequest, SwitchHandsResponse, 
    Notification, 
    DiceRequirementType, DiceType
)
import functools
import json
from log import logger
from google.protobuf.json_format import MessageToDict, MessageToJson
from pathlib import Path
from dice_pay import DicePay
import os
import re
import itertools
import threading
import queue
import time
import random
from copy import deepcopy
from aiapisync import LLMAPI
from dotenv import load_dotenv
import random
load_dotenv()



normal_dice_requirement_type = [DiceRequirementType.DICE_REQ_CRYO, DiceRequirementType.DICE_REQ_HYDRO, DiceRequirementType.DICE_REQ_PYRO, DiceRequirementType.DICE_REQ_ELECTRO, DiceRequirementType.DICE_REQ_ANEMO, DiceRequirementType.DICE_REQ_GEO, DiceRequirementType.DICE_REQ_DENDRO]
normal_dice_type = [DiceType.DICE_CRYO, DiceType.DICE_HYDRO, DiceType.DICE_PYRO, DiceType.DICE_ELECTRO, DiceType.DICE_ANEMO, DiceType.DICE_GEO, DiceType.DICE_DENDRO]

raw_json_action_cards = json.load(open('game_texts/action_cards.json'))
# fix 裂晶弹片 find the dict with id = 116081, and set obtainable to false
for action_card in raw_json_action_cards:
    if action_card["id"] == 116081:
        action_card["obtainable"] = False
raw_json_characters = json.load(open('game_texts/characters.json'))
raw_json_entities = json.load(open('game_texts/entities.json'))
# json_keywords = json.load(open('game_texts/keywords.json'))
raw_json_keywords = json.load(open('game_texts/neededkeywords.json'))
game_rules = Path('game_texts/game_rules.txt').read_text()

#TODO: rawxxx have id info, may better not to remove

def wash_characters(characters):
    remove_keys = ["shareId", "sinceVersion", "englishName", "storyTitle", "storyText", "cardFace", "icon"]
    remove_skill_keys = ["englishName", "rawDescription", "keyMap", "icon"]
    for character in characters:
        for key in remove_keys:
            character.pop(key, None)
        for skill in character["skills"]:
            for key in remove_skill_keys:
                skill.pop(key, None)
    return characters

def wash_action_cards(action_cards):
    remove_keys = ["shareId", "sinceVersion", "englishName", "storyTitle", "storyText", "rawDescription", "rawPlayingDescription", "cardFace", "icon"]
    for action_card in action_cards:
        for key in remove_keys:
            action_card.pop(key, None)
        for target in action_card["targetList"]:
            target.pop("rawHintText", None)
    return action_cards

def wash_entities(entities):
    remove_keys = ["shareId", "englishName", "rawDescription", "rawPlayingDescription", "cardFace", "buffIcon", "buffIconHash"]
    for entity in entities:
        for key in remove_keys:
            entity.pop(key, None)
    return entities

def wash_keywords(keywords):
    remove_keys = ["rawName", "rawDescription"]
    for keyword in keywords:
        for key in remove_keys:
            keyword.pop(key, None)
    return keywords

def wash_skills(skills):
    remove_skill_keys = ["englishName", "rawDescription", "keyMap", "icon"]
    for skill in skills:
        for key in remove_skill_keys:
            skill.pop(key, None)
    return skills

raw_json_skills = [skill for char in raw_json_characters for skill in char["skills"]]

raw_concat_json = raw_json_action_cards + raw_json_characters + raw_json_entities + raw_json_keywords + raw_json_skills

# These are not washed.
raw_id_mapping = {}
id_to_name = {}
name_to_id = {}
for card in raw_concat_json:
    raw_id_mapping[card["id"]] = card
    id_to_name[card["id"]] = card["name"]
    name_to_id[card["name"]] = card["id"]

json_characters = deepcopy(raw_json_characters)
json_action_cards = deepcopy(raw_json_action_cards)
json_entities = deepcopy(raw_json_entities)
json_keywords = deepcopy(raw_json_keywords)
json_skills = deepcopy(raw_json_skills)

wash_characters(json_characters)
wash_action_cards(json_action_cards)
wash_entities(json_entities)
wash_keywords(json_keywords)
wash_skills(json_skills)
concat_json = json_action_cards + json_characters + json_entities + json_keywords + json_skills

id_mapping = {}
for card in concat_json:
    id_mapping[card["id"]] = card


# for characters, collect all entity and cards. Then for all cards, collect cards and  entities.
# no need to collect characters A and keywords
# A char C entity/card S skill  K keyword 
def get_all_relevant_text(characters_id, action_cards_id):
    # get all skills rawDescription text. If cards or entities occur in that text, then add these info to that skill text. 
    
    chars = deepcopy([id_mapping[cid] for cid in characters_id]) # used to insert and return 
    sid_to_skill = {}
    for char in chars:
        for skill in char["skills"]:
            skill["relatedCardsAndEntities"] = []
            sid_to_skill[skill["id"]] = skill
    raw_chars = [raw_id_mapping[cid] for cid in characters_id] # read only
    raw_char_skills = [skill for char in raw_chars for skill in char["skills"]]
    
    for skill in raw_char_skills:
        # find $[C127033] in skill["rawDescription"] and get the c id.
        # insert the entity info to chars[skill]["description"]
        pattern = r'\$\[C(\d+)\]'
        # breakpoint()
        matches = re.findall(pattern, skill["rawDescription"])
        for match in matches:
            c_id = int(match)
            sid_to_skill[skill["id"]]["relatedCardsAndEntities"].append(deepcopy(id_mapping[c_id]))
            # 芙宁娜的cards不必二次插入entities


    
    cards = deepcopy([id_mapping[cid] for cid in action_cards_id]) # used to insert and return 
    # cards no need to skill, directly insert entities
    cid_to_card = {}
    for card in cards:
        cid_to_card[card["id"]] = card
        card["relatedCardsAndEntities"] = []
    raw_cards = [raw_id_mapping[cid] for cid in action_cards_id] # read only
    
    for card in raw_cards:
        # find $[C127033] in skill["rawDescription"] and get the c id.
        # insert the entity info to chars[skill]["description"]
        pattern = r'\$\[C(\d+)\]'
        matches = re.findall(pattern, card["rawDescription"])
        for match in matches:
            c_id = int(match)
            cid_to_card[card["id"]]["relatedCardsAndEntities"].append(deepcopy(id_mapping[c_id]))

            #entity对应cards就是瑟琳美露莘那种，应该不用二次插入cards
            #先不考虑美露莘这种了
    
    return chars, cards

@functools.lru_cache(maxsize=1000)
def simple_name_to_id(name):
    # first find name is a substring of any key in name_to_id.keys()
    for key in name_to_id.keys():
        if name in key:
            return name_to_id[key]
    return None


# 星柚
# AZFy20EQAUHC9UUQFVFB94QQCWJhBo8RClJxB5gRFGICCTEUDLLxi8AZDaJRDMYRDcEB
example_deck = {
    "characters": [
        1107,
        1507,
        1204
    ],
    "cards": [
        215071,
        215071,
        311206,
        311303,
        311406,
        312004,
        312004,
        312018,
        312018,
        321013,
        321013,
        322004,
        322004,
        322008,
        330001,
        332004,
        332004,
        332005,
        332006,
        332006,
        332021,
        332021,
        332022,
        332022,
        332024,
        332025,
        332032,
        332032,
        333003,
        333004
    ]
}

retain_cards = ["立本", "幻戏", "瓦格纳", "贯月", "运筹帷幄"]
retain_cards_id = [simple_name_to_id(card) for card in retain_cards]

DECK0 = Deck(characters=example_deck["characters"], cards=example_deck["cards"])
DECK1 = Deck(characters=example_deck["characters"], cards=example_deck["cards"])


# class decorator @action:
# if the method is decorated with @action, the funcname and return copy is saved to the action_history

def action(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return_value = func(self, *args, **kwargs)
        # self.action_history.append((func.__name__, return_value))
        # print(self.action_history[-1])
        return return_value
    return wrapper

class HeuristicPlayer(Player):

    def __init__(self, player_id):
        self.player_id = player_id        
        self.notifications = []
        self.action_history = []

    def get_last_notification(self):
        return self.notifications[-1]
    
    def get_current_player(self):
        return self.get_last_notification().state.player[self.player_id]

    def get_dice(self):
        return self.get_current_player().dice
    
    def get_hand_cards(self):
        return list(self.get_current_player().hand_card)

    def on_notify(self, notification: Notification):
        self.notifications.append(notification)

    def on_io_error(self, error_msg: str):
        # Handle IO errors
        print(f"IO Error: {error_msg}")
        print(self.action_history[-1])
        print(self.get_dice())

    @action
    def on_choose_active(self, request: ChooseActiveRequest) -> ChooseActiveResponse:
        # Choose the first available active character
        if request.candidate_ids:
            chosen_id = request.candidate_ids[0]
            return ChooseActiveResponse(active_character_id=chosen_id)
        else:
            return ChooseActiveResponse(active_character_id=None)
    
    def element_to_dice(self, element):
        # first get element string from element
        element = element.split("_")[-1]
        element = "DICE_" + element
        if element == "DICE_CRYO":
            return DiceType.DICE_CRYO
        elif element == "DICE_HYDRO":
            return DiceType.DICE_HYDRO
        elif element == "DICE_PYRO":
            return DiceType.DICE_PYRO
        elif element == "DICE_ELECTRO":
            return DiceType.DICE_ELECTRO
        elif element == "DICE_ANEMO":
            return DiceType.DICE_ANEMO
        elif element == "DICE_GEO":
            return DiceType.DICE_GEO
        elif element == "DICE_DENDRO":
            return DiceType.DICE_DENDRO
        else:
            return DiceType.DICE_UNSPECIFIED
        
    def get_all_characters(self):
        return self.get_current_player().character

    def get_current_character(self):
        all_characters = self.get_all_characters()
        current_character = [i for i in all_characters if i.id == self.get_current_player().active_character_id][0]
        return current_character
    
    #TODO: not consider multiple element characters
    def character_to_element(self, definition_id):
        tag = [tag for tag in id_mapping[definition_id]["tags"] if tag.startswith("GCG_TAG_ELEMENT_")][0]
        characters_element = self.element_to_dice(tag)
        return characters_element

    def all_character_element(self):
        return [self.character_to_element(character.definition_id) for character in self.get_all_characters()]

    def current_character_element(self):
        return self.character_to_element(self.get_current_character().definition_id)

    @action
    def on_reroll_dice(self, request: RerollDiceRequest) -> RerollDiceResponse:
        retain_dice = set(self.all_character_element() + [DiceType.DICE_OMNI])
        dice = self.get_dice()
        dice_to_reroll = [d for d in dice if d not in retain_dice]
        return RerollDiceResponse(dice_to_reroll=dice_to_reroll)

    @action
    def on_select_card(self, request: SelectCardRequest) -> SelectCardResponse:
        # Select the first available card
        # print(request.candidate_definition_ids)
        if request.candidate_definition_ids:
            selected_id = request.candidate_definition_ids[0]
            return SelectCardResponse(selected_definition_id=selected_id)
        else:
            return SelectCardResponse(selected_definition_id=None)

    @action
    def on_switch_hands(self, request: SwitchHandsRequest) -> SwitchHandsResponse:
        switch_ids = []
        hand_cards = self.get_hand_cards()
        for i, card in enumerate(hand_cards):
            if card.definition_id not in retain_cards_id:
                switch_ids.append(hand_cards[i].id)
        return SwitchHandsResponse(removed_hand_ids=switch_ids)
    
    #WARNING: use color assert
    def is_able_to_pay(self, required_cost):
        current_unpaid_dice = [i for i in self.get_dice()]
        all_req_type = [req.type for req in required_cost]
        # assert if aligned occurs, then void does not occur
        assert not (DiceRequirementType.DICE_REQ_ALIGNED in all_req_type and DiceRequirementType.DICE_REQ_VOID in all_req_type)
        # assert if aligned occurs, then normal dice does not occur
        assert not (DiceRequirementType.DICE_REQ_ALIGNED in all_req_type and any(req.type in normal_dice_requirement_type for req in list(required_cost)))
        # sort required_cost as DICE_REQ_ENERGY, DICE_REQ_LEGEND, DICE_REQ_ALIGNED, normal_dice_requirement_type, DICE_REQ_VOID, big to small
        required_cost = sorted(list(required_cost), key=lambda x: x.type, reverse=True)
        for req in required_cost:
            # TODO: have not found place of checking whether legend is used or energy
            if req.type == DiceRequirementType.DICE_REQ_LEGEND:
                if self.get_current_player().legend_used:
                    return False
            elif req.type == DiceRequirementType.DICE_REQ_ENERGY:
                if not self.get_current_character().energy >= req.count:
                    return False
            elif req.type == DiceRequirementType.DICE_REQ_ALIGNED:
                # check the max count of the element in current_unpaid_dice (omni is not counted)
                max_count = max([current_unpaid_dice.count(i) for i in normal_dice_type])
                return max_count + current_unpaid_dice.count(DiceType.DICE_OMNI) >= req.count
            # FIXME dice num and omni
            elif req.type in normal_dice_requirement_type:
                # Calculate the number of dice of the current type and the number of omni dice
                type_count = current_unpaid_dice.count(req.type)
                omni_count = current_unpaid_dice.count(DiceType.DICE_OMNI)
                
                # If the normal dice are not enough, omni dice need to be used to supplement
                if type_count < req.count:
                    needed_omni = req.count - type_count
                    if omni_count < needed_omni:
                        return False
                    
                    # Remove the used dice
                    for _ in range(type_count):
                        current_unpaid_dice.remove(req.type)
                    for _ in range(needed_omni):
                        current_unpaid_dice.remove(DiceType.DICE_OMNI)
                else:
                    # If the normal dice are enough, use them directly
                    for _ in range(req.count):
                        current_unpaid_dice.remove(req.type)
            elif req.type == DiceRequirementType.DICE_REQ_VOID:
                return len(current_unpaid_dice) >= req.count
            else:
                raise ValueError(f"Unknown dice requirement type: {req.type}")
        return True
    
    #WARNING: use color assert
    def smart_pay(self, required_cost):
        # still needs to ensure if element tuning, cannot use omni dice or active character's element
        # only care about dice. 
        # also need to meet the align void not occur at the same time assert, but no need to check.

        # if require aligned, then find the max count in not important normal dice. 
        # If that num + omni is enough, then pay with that + omni.
        # If not, then pay with important normal dice that is not omni.

        # First add required normal dice, 
        # Then if has void, then pay from the most not important to most important.
        # Important order: omni, character element normal dice, other normal dice

        # return the dice to pay

        current_dice = [i for i in self.get_dice()]
        character_elements = self.all_character_element()
        # make current character elements be the last, so when doing for loop, always be the last to want to pay.
        # This also ensure if element tuning, cannot use omni dice or active character's element
        current_character_element = self.current_character_element()
        character_elements.remove(current_character_element)
        character_elements.append(current_character_element)
        used_dice = []
        
        # required_cost = sorted(list(required_cost), key=lambda x: x.type, reverse=True) # why I need this?

        # Process normal element requirements
        for req in required_cost:
            if req.type in normal_dice_requirement_type:
                req_dice = req.type #implicit transform req.type to DiceType
                for _ in range(req.count):
                    if req_dice in current_dice:
                        used_dice.append(req_dice)
                        current_dice.remove(req_dice)
                    elif DiceType.DICE_OMNI in current_dice:
                        used_dice.append(DiceType.DICE_OMNI)
                        current_dice.remove(DiceType.DICE_OMNI)

        # Process ALIGNED requirements
        # actually assert no two aligned at the same time
        aligned_req = next((req for req in required_cost if req.type == DiceRequirementType.DICE_REQ_ALIGNED), None)
        if aligned_req:
            # Count the number of each non-character element dice
            dice_counts = {}
            for d in current_dice:
                if d in normal_dice_type and d not in character_elements:
                    dice_counts[d] = dice_counts.get(d, 0) + 1
            
            # Find the most common non-character element dice
            most_common_dice = None
            max_count = 0
            for dice_type, count in dice_counts.items():
                if count > max_count:
                    max_count = count
                    most_common_dice = dice_type
            
            omni_count = current_dice.count(DiceType.DICE_OMNI)
            
            if most_common_dice and (max_count + omni_count >= aligned_req.count):
                # Prefer to use normal dice
                for _ in range(min(max_count, aligned_req.count)):
                    used_dice.append(most_common_dice)
                    current_dice.remove(most_common_dice)
                
                # Supplement with the required omni dice
                remaining = aligned_req.count - len(used_dice)
                for _ in range(remaining):
                    used_dice.append(DiceType.DICE_OMNI)
                    current_dice.remove(DiceType.DICE_OMNI)
            else:
                # If there are not enough non-character element dice, try to use character element dice
                for element in character_elements:
                    element_count = current_dice.count(element)
                    if element_count + omni_count >= aligned_req.count:
                        # Use this character element's dice
                        for _ in range(min(element_count, aligned_req.count)):
                            used_dice.append(element)
                            current_dice.remove(element)
                        
                        # Supplement with the required omni dice
                        remaining = aligned_req.count - len(used_dice)
                        for _ in range(remaining):
                            used_dice.append(DiceType.DICE_OMNI)
                            current_dice.remove(DiceType.DICE_OMNI)
                        break

        # Process VOID requirements
        void_req = next((req for req in required_cost if req.type == DiceRequirementType.DICE_REQ_VOID), None)
        if void_req:
            remaining_count = void_req.count
            
            # Prefer to use non-character element normal dice
            non_character_dice = [d for d in current_dice if d in normal_dice_type and d not in character_elements]
            for dice in non_character_dice:
                if remaining_count > 0:
                    used_dice.append(dice)
                    current_dice.remove(dice)
                    remaining_count -= 1
            
            # Then use character element dice
            character_dice = [d for d in character_elements for _ in range(current_dice.count(d))]
            for dice in character_dice:
                if remaining_count > 0:
                    used_dice.append(dice)
                    current_dice.remove(dice)
                    remaining_count -= 1
            
            # Finally use omni dice
            while remaining_count > 0 and DiceType.DICE_OMNI in current_dice:
                used_dice.append(DiceType.DICE_OMNI)
                current_dice.remove(DiceType.DICE_OMNI)
                remaining_count -= 1
        
        return used_dice
    
    #WARNING: use color assert
    #This code may be used very later, and then 
    #不考虑鲸鱼的优先万能支付
    # WARNING: deprecated!!!!! Now want to use boby new ver
    def smart_all_pay(self, required_cost):
        # return all possible smart pay plan (using poppy edge cut), [[DiceType]]
        # still better to copy mine for normal and aligned. If meet void, then use poppy.
        
        def index_plan_to_deduplicate_dice_plan(index_plan):
            deduplicate_index_plan = [list(plan) for plan in set(tuple(plan) for plan in index_plan)]
            deduplicate_dice_plan = [[current_dice[i] for i in d] for d in deduplicate_index_plan]
            return deduplicate_dice_plan

        current_dice = [i for i in self.get_dice()]
        character_elements = self.all_character_element()
        powerset_character_elements = list(itertools.chain.from_iterable(itertools.combinations(character_elements, r) for r in range(len(character_elements) + 1)))

        if (req := next((r for r in required_cost if r.type == DiceRequirementType.DICE_REQ_ALIGNED), None)):
            used_dice = []
            for try_keep in powerset_character_elements:
                dice_pay_0 = DicePay(current_dice)
                for paypolicy in [0, 1]:
                    dice_pay_1 = deepcopy(dice_pay_0)
                    used_dice.append(dice_pay_1.get_dice_aligned(req.count, policy=paypolicy, try_keep=try_keep))
            return index_plan_to_deduplicate_dice_plan(used_dice)

        
        if (req := next((r for r in required_cost if r.type in normal_dice_requirement_type), None)):
            used_dice = []
            for try_keep in powerset_character_elements:
                dice_pay_0 = DicePay(current_dice)
                for paypolicy in [0, 1]:
                    dice_pay_1 = deepcopy(dice_pay_0)
                    dice_pay_1.get_dice_element(req.type, req.count)
                    if (req2 := next((r for r in required_cost if r.type == DiceRequirementType.DICE_REQ_VOID), None)):
                        for paypolicy2 in [0, 1]:
                            for try_keep2 in powerset_character_elements:
                                dice_pay_2 = deepcopy(dice_pay_1)
                                used_dice.append(dice_pay_2.get_dice_void(req2.count, policy=paypolicy2, try_keep=try_keep2))
                    else:
                        used_dice.append(dice_pay_1.get_dice_element(req.type, req.count))
            return index_plan_to_deduplicate_dice_plan(used_dice)
        
        if (req := next((r for r in required_cost if r.type == DiceRequirementType.DICE_REQ_VOID), None)):
            for paypolicy2 in [0, 1]:
                for try_keep2 in powerset_character_elements:
                    dice_pay_2 = DicePay(current_dice)
                    used_dice.append(dice_pay_2.get_dice_void(req.count, policy=paypolicy2, try_keep=try_keep2))
            return index_plan_to_deduplicate_dice_plan(used_dice)

        return []
    
    def on_valid_action(self, actions) -> int:
        # return the index of the action that is valid
        # priority q e playcard a end no switch_character and no element_tuning
        """Return index of highest priority valid action"""

        
        # Define action priorities (higher number = higher priority)
        PRIORITY = {
            "burst": 5,      # Q技能
            "skill": 4,      # E技能
            "card": 3,       # 打出手牌
            "attack": 2,     # 普通攻击
            "end": 1,        # 回合结束
            "switch": 0,     # 切换角色
            "tuning": 0      # 元素调和
        }
        
        def get_action_type(action) -> str:
            """Classify action into one of the priority types"""
            if action.HasField("elemental_tuning"):
                return "tuning"
            if action.HasField("declare_end"):
                return "end"
            if action.HasField("switch_active"):
                return "switch"
                
            # For skill actions, check skill type
            if action.HasField("use_skill"):
                skill_type = id_mapping[action.use_skill.skill_id]["type"]
                if "GCG_SKILL_TAG_Q" in skill_type:
                    return "burst"
                if "GCG_SKILL_TAG_E" in skill_type:
                    return "skill"
                if "GCG_SKILL_TAG_A" in skill_type:
                    return "attack"
                    
            # For card actions
            if action.HasField("play_card"):
                return "card"
                
            return "end"  # Default to lowest priority
        
        # Sort actions by priority and return highest priority valid action
        action_priorities = [(i, PRIORITY[get_action_type(action)]) 
                            for i, action in enumerate(actions)]
        # logger.info("Notification: " + str(self.get_last_notification()))
        # logger.info("Actions: " + str(actions))
        # logger.info("Action Priorities: " + str(action_priorities))
        # logger.info("Chosen Index: " + str(max(action_priorities, key=lambda x: x[1])[0]))
        return max(action_priorities, key=lambda x: x[1])[0]
    

    @action
    def on_action(self, request: ActionRequest) -> ActionResponse:
        used_dice: list[DiceType] = []
        actions = list(enumerate(request.action))
        valid_actions = []
        valid_actions_index = [] # only used to translate index in return
        for i, action in actions:
            if action.HasField("elemental_tuning"):
                #if all omni or front character element, then continue
                if all(d == DiceType.DICE_OMNI or d == self.current_character_element() for d in self.get_dice()):
                    continue
                valid_actions.append(action)
                valid_actions_index.append(i)
                continue
            if self.is_able_to_pay(action.required_cost):
                valid_actions.append(action)
                valid_actions_index.append(i)
                continue
        chosen_index = self.on_valid_action(valid_actions)
        
        if self.player_id == 0:
            logger.info(f"dice: {self.get_dice()}")
            logger.info(f"required_cost: {valid_actions[chosen_index].required_cost}")
            logger.info(f"character_element: {self.all_character_element()}")
            # logger.info(f"smart_all_pay: {self.smart_all_pay(valid_actions[chosen_index].required_cost)}")
            logger.info(f"smart_pay: {self.smart_pay(valid_actions[chosen_index].required_cost)}")
            # print("chosen_index", chosen_index)
        used_dice = self.smart_pay(valid_actions[chosen_index].required_cost)
        self.action_history.append((valid_actions[chosen_index], used_dice))
        return ActionResponse(chosen_action_index=valid_actions_index[chosen_index], used_dice=used_dice)

class LanguageBasedPlayer(HeuristicPlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
    def process_state_action(self, state, actions):
        json_state = MessageToDict(state)
        relevant_chars, relevant_cards = get_all_relevant_text(example_deck["characters"], example_deck["cards"])
        id_to_relevant_char = {}
        id_to_relevant_card = {}
        for c in relevant_chars:
            id_to_relevant_char[c["id"]] = c
        for c in relevant_cards:
            id_to_relevant_card[c["id"]] = c
        for p in json_state["state"]["player"]:
            for c in p["character"]:
                c["details"] = id_to_relevant_char[c["definitionId"]]
        for p in json_state["state"]["player"]:
            if "handCard" in p:     
                for c in p["handCard"]:
                    if "definitionId" in c:
                        c["details"] = id_to_relevant_card[c["definitionId"]]
        json_state["state"]["player"] = {"my_info": json_state["state"]["player"][self.player_id], "opponent_info": json_state["state"]["player"][(self.player_id + 1) % 2]}
        
        json_actions = [MessageToDict(json_action) for json_action in actions]
        # for json_action in json_actions:
            # if "play_card" in json_action.keys():
                # json_action["play_card"]["details"] = id_to_relevant_card[json_action["play_card"]["cardId"]] #duplication
        json_actions = [("choice_" + str(i), json_action) for i, json_action in enumerate(json_actions)]
        return json_state, json_actions

class LLMPlayer(LanguageBasedPlayer):
    def __init__(self, player_id):
        super().__init__(player_id)

        # llm_api = LLMAPI(script_args.score_model, script_args.useapi, script_args.api_key,
                        # script_args.requests_per_minute)
        # llm_api = LLMAPI("Qwen/Qwen2.5-72B-Instruct-Turbo", 'together', os.getenv("TOGETHER_API_KEY"), 300)
        # if player_id == 0:
            # self.llm_api = LLMAPI("gpt-4o-mini", 'openai', os.getenv("OPENAI_API_KEY"), 300)
        # else:
        self.llm_api = LLMAPI("gpt-4o-mini", 'openai', os.getenv("OPENAI_API_KEY"), 300)
        self.get_api_response = self.llm_api.get_api_response

    def on_valid_action(self, actions) -> int:
        state = self.get_last_notification()
        json_state, json_actions = self.process_state_action(state, actions)

        for action in json_actions:
            self.q(json_state, action)

        state_action_prompt = f"""
        Here is the current state:
        {json_state}

        Here is the current possible actions to take:
        {json_actions}
        """

        build_prompt = f"""
        I'm playing a card game. I want you to decide which action is the best to take. Think step by step and tell me your plan and reason of each thought.
        Here is the rule of the game:
        {game_rules}
        Here is the basic keywords:
        {json_keywords}

        {state_action_prompt}

        Now, I want you to decide which action is the best to take. Think step by step and tell me your plan and reason of each thought.
        Finally, tell me the index of the action you choose, by ending with "choice_{{i}}", where i is the index number of the action you choose.
        """
        # logger.info("build_prompt: " + build_prompt)
        llm_ans = self.get_api_response(build_prompt)
        if self.player_id == 1:
            logger.info("####BUILD_PROMPT: \n" + build_prompt)
            logger.info("####LLM_ANS: \n" + llm_ans)

        def extract_choice_number(text: str) -> int:
            """Extract the number after the last 'choice_' in text.
            
            Args:
                text (str): Input text
                
            Returns:
                int: The extracted number, or None if no valid number found
            """
            try:
                get_last_index = text.rfind("choice_")
                if get_last_index == -1:
                    return None
                    
                start_index = get_last_index + 7
                if start_index >= len(text):
                    return None
                    
                # Don't initialize num_index_end until after we find a non-digit
                for i in range(start_index, len(text)):
                    if not text[i].isdigit():
                        return int(text[start_index:i])
                        
                # If we get here, the number goes to the end of the string
                return int(text[start_index:])
                
            except Exception as e:
                breakpoint()
                logger.error(f"Error extracting choice number: {e}")
                return None
        
        chosen_index = extract_choice_number(llm_ans)
        print("chosen_index", chosen_index)
        return chosen_index


    def q(self, state, action) -> int:
        build_prompt = f"""
        I'm playing a card game. I want you to decide after I choose the action, what's the winning probability of the game. Output the probability with a number between 0 and 1, with 2 decimal places.
        Here is the rule of the game:
        {game_rules}
        Here is the basic keywords:
        {json_keywords}

        Here is the current state:
        {state}

        Here is the action I choose:
        {action}

        Now, I want you to I want you to decide after I choose the action, what's the winning probability of the game. Output the probability with a number between 0 and 1, with 2 decimal places. Directly output the number.
        """

        def extract_number(text: str) -> float:
            """Extract the number in text.
            
            Args:
                text (str): Input text
                
            Returns:
                fload: The extracted number between 0 and 1, or None if no valid number found
            """
            pattern = r'\d+\.\d{2}'
            match = re.search(pattern, text)
            if match:
                return float(match.group(0))
            return None
        
        lm_ans = self.get_api_response(build_prompt)
        logger.info(lm_ans, log_file="lm_ans.log")
        predicted_winning_probability = extract_number(lm_ans)
        print("predicted_winning_probability", predicted_winning_probability)

        return predicted_winning_probability




from model import StreamingDataset, train, get_lm_response, reason, prepare_model

class LMTrainer:
    def __init__(self):
        self.max_size = 200 # max size of the queue
        self.data_queue = queue.Queue(maxsize=self.max_size) # This queue will be fetched by the player class and put into the data

        # Start the consumer thread
        self.run_mode = ["train_loss_head", "train_rl"][0]
        self.consumer_thread = threading.Thread(target=self.train_model)
        self.model = prepare_model(self.run_mode)
        self.stream_dataset = None

    def start(self):
        self.consumer_thread.start()
    
    def stop(self):
        self.consumer_thread.join()

    def train_model(self):
        self.stream_dataset = StreamingDataset(self.consume_data)
        train(self.stream_dataset, self.run_mode, self.model)
    
    def want_to_stop(self):
        if self.stream_dataset is None:
            return False
        return self.stream_dataset.want_to_stop

    def consume_data(self):
        while self.data_queue.empty():
            print(f"No enough data to consume, waiting for more data...")
            time.sleep(10)
        data = self.data_queue.get()
        self.data_queue.task_done() # FIXME: though I haven't complete the data, I still need to mark it as done
        return data


class LMPlayer(LanguageBasedPlayer):
    def __init__(self, player_id, lmtrainer: LMTrainer):
        super().__init__(player_id)
        self.sapath = [] #[(s,a),...]
        self.epsilon = 0.05
        self.lmtrainer = lmtrainer
        self.data_queue = lmtrainer.data_queue
        self.run_mode = lmtrainer.run_mode
        self.model = lmtrainer.model

    def on_notify(self, notification: Notification):
        self.notifications.append(notification)
        if notification.state.HasField("winner"):
            self.on_game_end(notification.state.winner)

    def produce_data(self, data):
        # it is called after a data is produced. The producing process use a model to produce the data.
        # put data into the queue. If the queue is full, wait until the queue is not full
        s, a, r = data
        while self.data_queue.full():
            print("Queue is full, waiting to produce data...")
            time.sleep(0.02)
        # Put data into the queue
        self.data_queue.put((self.build_sa_prompt(s,a), r))

    def on_game_end(self, game_result):
        if self.run_mode == "train_rl":
            for s,a in self.sapath:
                self.produce_data((s, a, game_result))

    def on_valid_action(self, actions) -> int:
        state = self.get_last_notification()
        json_state, json_actions = self.process_state_action(state, actions)
        if random.random() < self.epsilon:
            # random choose an action
            chosen_index = random.randint(0, len(json_actions) - 1)
        else:
            max_q, max_q_action = -100, None
            if self.run_mode == "train_loss_head":
                q_list = self.q_from_casual_lm(state, json_actions)
                for action, q in zip(json_actions, q_list):
                    if q != -100:
                        self.produce_data((json_state, action, q))
            elif self.run_mode == "train_rl":
                q_list = self.q(state, json_actions)
            else:
                raise ValueError("Invalid run mode")
            max_q = max(q_list)
            if max_q == -100:
                logger.info("All is bad, random choose an action", log_file="no_valid_number_found.log")
                chosen_index = random.randint(0, len(json_actions) - 1)
            else:
                chosen_index = q_list.index(max_q)
        chosen_action = json_actions[chosen_index]
        self.sapath.append((json_state, chosen_action))
        return chosen_index

    def build_sa_prompt(self, state, action):
        build_prompt = f"""
        I'm playing a card game. I want you to decide after I choose the action, what's the winning probability of the game. Output the probability with a number between 0 and 1, with 2 decimal places.
        Here is the rule of the game:
        {game_rules}
        Here is the basic keywords:
        {json_keywords}

        Here is the current state:
        {state}

        Here is the action I choose:
        {action}

        Now, I want you to I want you to decide after I choose the action, what's the winning probability of the game. Output the probability with a number between 0 and 1, with 2 decimal places. Directly output the number.
        """
        return build_prompt


    def q_from_casual_lm(self, state, actions) -> float:
        build_prompts = [self.build_sa_prompt(state, action) for action in actions]
        def extract_number(text: str) -> float:
            """Extract the number in text.
            
            Args:
                text (str): Input text
                
            Returns:
                fload: The extracted number between 0 and 1, or None if no valid number found
            """
            pattern = r'\d+\.\d{2}'
            match = re.search(pattern, text)
            if match:
                ret = float(match.group(0))
                logger.info("Valid number" + str(ret) + " found in text: " + text, log_file="valid_number_found.log")
                return ret
            logger.info("No valid number found in text: " + text, log_file="no_valid_number_found.log")
            return -100
        
        lm_ans = get_lm_response(build_prompts)
        logger.info("lm_ans: " + str(lm_ans), log_file="lm_ans.log")
        predicted_winning_probabilities = [extract_number(lm_ans) for lm_ans in lm_ans]
        print("predicted_winning_probability", predicted_winning_probabilities)

        return predicted_winning_probabilities

    def q(self, state, actions) -> float:
        build_prompts = [self.build_sa_prompt(state, action) for action in actions]
        lm_ans = reason(build_prompts, self.model)
        return lm_ans

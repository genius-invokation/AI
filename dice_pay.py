from enum import Enum
from collections import defaultdict

class DiceRequirementType(Enum):
    DICE_REQ_VOID = 0
    DICE_REQ_CRYO = 1
    DICE_REQ_HYDRO = 2
    DICE_REQ_PYRO = 3
    DICE_REQ_ELECTRO = 4
    DICE_REQ_ANEMO = 5
    DICE_REQ_GEO = 6
    DICE_REQ_DENDRO = 7
    DICE_REQ_ALIGNED = 8
    DICE_REQ_ENERGY = 9
    DICE_REQ_LEGEND = 10

class DiceType(Enum):
    DICE_UNSPECIFIED = 0
    DICE_CRYO = 1
    DICE_HYDRO = 2
    DICE_PYRO = 3
    DICE_ELECTRO = 4
    DICE_ANEMO = 5
    DICE_GEO = 6
    DICE_DENDRO = 7
    DICE_OMNI = 8

class KeepPolicy(Enum):
    KEEP_DEFAULT = 0
    KEEP_DOUBLE = 1
    KEEP_TRIPLE = 2

class PayPolicy(Enum):
    KEEP_DIFF = 0
    KEEP_SAME = 1

class DicePay:

    def __init__(self, curr_dice):
        self.dice = defaultdict(list)
        for i, d in enumerate(curr_dice):
            self.dice[d].append(i)
        self.dice = dict(self.dice) 
    
    def get_dice_element(self, req_type, count):
        dice_resp = []
        if len(self.dice.get(req_type, [])) >= count:
            dice_resp.extend(self.dice.get(req_type, [])[:count])
        else:
            dice_resp.extend(self.dice.get(req_type, []))
            if len(self.dice.get(DiceType.DICE_OMNI.value, [])) >= count - len(dice_resp):
                dice_resp.extend(self.dice.get(DiceType.DICE_OMNI.value, [])[len(dice_resp) - count:])
        if len(dice_resp) == count:
            return dice_resp
        return None

    def get_dice_void(self, count, policy, try_keep):
        """
        Get dice indices based on the given count, policy, and try_keep list.

        :param count: Number of dice needed.
        :param policy: Payment policy (KEEP_SAME or KEEP_DIFF).
        :param try_keep: List of dice types to prioritize keeping.
        :return: A list of dice indices to "pay", or None if not enough dice.
        """
        # 骰子数量不够
        if len(self) < count:
            return None

        pay_index = []

        if policy == PayPolicy.KEEP_SAME.value:
            # 区分有效骰和杂色骰
            pay_dice = {k: v for k, v in self.dice.items() if k not in try_keep}
            try_keep_dice = {k: v for k, v in self.dice.items() if k in try_keep}
            omni = []
            # 按骰子数量从少到多排序并收集
            for dice_set in (pay_dice, try_keep_dice):
                for dice_type, indices in sorted(dice_set.items(), key=lambda item: len(item[1])):
                    if dice_type == DiceType.DICE_OMNI.value:
                        omni.extend(indices)
                        continue
                    pay_index.extend(indices)
                    if len(pay_index) >= count:
                        return pay_index[:count]
            pay_index.extend(omni)
            return pay_index[:count]

        elif policy == PayPolicy.KEEP_DIFF.value:
            # 分优先级排列骰子
            priorities = {"prior1": [], "prior2": [], "prior3": [], "prior4": [], "omni": []}

            for k, v in self.dice.items():
                if len(v) >= 2:
                    if k == DiceType.DICE_OMNI.value:
                        priorities["omni"].extend(v)
                    elif k in try_keep:
                        priorities["prior3"].extend(v[:-2])
                        priorities["prior4"].append(v[-1])
                    else:
                        priorities["prior1"].extend(v[:-2])
                        priorities["prior2"].append(v[-1])
                else:
                    if k == DiceType.DICE_OMNI.value:
                        priorities["omni"].extend(v)
                    elif k in try_keep:
                        priorities["prior4"].extend(v)
                    else:
                        priorities["prior2"].extend(v)

            # 按优先级收集骰子
            for key in ["prior1", "prior2", "prior3", "prior4", "omni"]:
                pay_index.extend(priorities[key])
                if len(pay_index) >= count:
                    return pay_index[:count]

    def get_dice_aligned(self, count, policy, try_keep):
        """
           Get aligned dice indices based on the given count, policy, and try_keep list.

           :param count: Number of dice needed.
           :param policy: Payment policy (KEEP_SAME or KEEP_DIFF).
           :param try_keep: List of dice types to prioritize keeping.
           :return: A list of dice indices to "pay", or None if not enough dice.
        """
        pay_index = []
        pay_dice = {}
        try_keep_dice = {}
        omni_dice = []
        for k, v in self.dice.items():
            if k == DiceType.DICE_OMNI.value:
                omni_dice = v
            elif k in try_keep:
                try_keep_dice[k] = v
            else:
                pay_dice[k] = v
        sorted_pay_dice = sorted(pay_dice.items(), key=lambda item: len(item[1]))
        sorted_try_keep_dice = sorted(try_keep_dice.items(), key=lambda item: len(item[1]))
        # 留同色，按骰子数量从少到多收集
        if policy == PayPolicy.KEEP_SAME.value:
            for k_v in sorted_pay_dice:
                pay_index.extend(k_v[1])
                if len(k_v[1]) >= count:
                    return k_v[1][:count]
                pay_index = []
            for k_v in sorted_try_keep_dice:
                pay_index.extend(k_v[1])
                if len(pay_index) >= count:
                    return pay_index[:count]
                pay_index = []
        # 留异色，按骰子数量从多到少收集
        if policy == PayPolicy.KEEP_DIFF.value:
            for k_v in reversed(sorted_pay_dice):
                pay_index.extend(k_v[1])
                if len(k_v[1]) >= count:
                    return k_v[1][:count]
                pay_index = []
            for k_v in reversed(sorted_try_keep_dice):
                pay_index.extend(k_v[1])
                if len(pay_index) >= count:
                    return pay_index[:count]
                pay_index = []
        # 单色骰子不够，万能支援下
        pay_index.extend(sorted_pay_dice[-1][1])
        pay_index.extend(omni_dice)
        if len(pay_index) >= count:
            return pay_index[:count]
        pay_index = []
        if len(sorted_try_keep_dice) > 0:
            pay_index.extend(sorted_try_keep_dice[-1][1])
            pay_index.extend(omni_dice)
            if len(pay_index) >= count:
                return pay_index[:count]
        return None

    def __len__(self):
        default_len = 0
        for k, v in self.dice.items():
            if k != DiceType.DICE_UNSPECIFIED:
                default_len += len(v)
        return default_len


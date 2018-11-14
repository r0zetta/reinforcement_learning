import random
import numpy as np
from collections import Counter

#pick_up_actions = [["picked", "up", "the", "OBJ"], ["got", "the", "OBJ"], ["acquired", "the", "OBJ"], ["grabbed", "the", "OBJ"]]
#put_down_actions = [["put", "the", "OBJ", "down"], ["left", "the", "OBJ", "there"], ["dropped", "the", "OBJ", "there"]]
#to_the_actions = ["went", "walked", "traveled", "strolled", "ran", "hurried"]
#observe_actions = ["saw","observed","noticed"]
#names = ["andy", "bill", "george", "matt", "gary", "bob", "john", "pete", "simon", "mark", "mary", "sue", "ann", "tracy", "lucy", "joanna", "katie", "louise", "susan", "jane"]
#places = ["bedroom", "bathroom", "kitchen", "garden", "hall", "yard", "basement", "cellar", "shop", "park", "office"]
#objects = ["ball", "book", "lamp", "laptop", "phone", "plate", "chair", "bottle", "spoon"]

names = ["george", "matt", "gary", "lucy", "joanna", "katie"]
places = ["house", "shop", "park", "office", "town", "seaside"]
objects = ["ball", "book", "phone", "plate", "bottle", "glass"]
observe_actions = ["saw"]
to_the_actions = ["went"]
pick_up_actions = [["took", "the", "OBJ"]]
put_down_actions = [["left", "the", "OBJ"]]
question = ["there", "where", "is", "to", "a", "in", "who", "has", "?"]


class game:
    def __init__(self, num_objects, num_names, num_moves):
        self.num_objects = num_objects
        self.num_names = num_names
        self.num_moves = num_moves
        self.max_sent_len = 6
        self.make_vocab()
        self.reset()

    def make_vocab(self):
        vocab = []
        for n in question:
            if n not in vocab:
                vocab.append(n)
        for n in names:
            if n not in vocab:
                vocab.append(n)
        for n in places:
            if n not in vocab:
                vocab.append(n)
        for n in objects:
            if n not in vocab:
                vocab.append(n)
        for n in to_the_actions:
            if n not in vocab:
                vocab.append(n)
        for n in observe_actions:
            if n not in vocab:
                vocab.append(n)
        for r in pick_up_actions:
            for n in r:
                if n not in vocab:
                    vocab.append(n)
        for r in put_down_actions:
            for n in r:
                if n not in vocab:
                    vocab.append(n)
        self.voc = {}
        self.voc["PAD"] = 0
        self.voc_inv = {}
        self.voc_inv[0] = "PAD"
        for index, word in enumerate(vocab):
            self.voc[word] = index+1
            self.voc_inv[index+1] = word
        self.vocab_size = len(self.voc)

    def reset(self):
        self.make_scenario()
        self.vectorize()
        self.state = self.scenario
        self.num_actions = self.vocab_size

    def vectorize_one_hot_2d(self):
        self.scenario_len = len(self.sent)
        self.scenario = np.zeros((self.scenario_len, self.max_sent_len, self.vocab_size), dtype=int)
        l = self.max_sent_len
        for sent_num, sentence in enumerate(self.sent):
            for word_pos, word in enumerate(sentence):
                self.scenario[sent_num][word_pos][self.voc[word]] = 1

    def vectorize_one_hot_flat(self):
        self.scenario_len = len(self.sent)
        self.scenario = np.zeros((self.scenario_len, self.max_sent_len * self.vocab_size), dtype=int)
        l = self.max_sent_len
        for sent_num, sentence in enumerate(self.sent):
            for word_pos, word in enumerate(sentence):
                self.scenario[sent_num][l*word_pos+self.voc[word]] = 1

    def vectorize_numerical(self):
        self.scenario_len = len(self.sent)
        self.scenario = np.zeros((self.scenario_len, self.max_sent_len), dtype=int)
        for sent_num, sentence in enumerate(self.sent):
            for word_pos, word in enumerate(sentence):
                self.scenario[sent_num][word_pos] = self.voc[word]

    def vectorize(self):
        self.answer_vec = np.zeros((self.vocab_size), dtype=int)
        self.answer_vec[self.voc[self.answer]] = 1
        self.answer_index = self.voc[self.answer]
        self.vectorize_one_hot_2d()

    def make_scenario(self):
        self.sent = []
        self.names = []
        while len(self.names) < self.num_names:
            c = random.choice(names)
            if c not in self.names:
                self.names.append(c)
        self.objects = []
        self.objects_seen = []
        self.people_seen = []
        while len(self.objects) < self.num_objects:
            c = random.choice(objects)
            if c not in self.objects:
                self.objects.append(c)
        self.trajectories = {}
        self.people_locations = {}
        self.places = []
        self.carrying = {}
        for name in self.names:
            self.people_locations[name] = 0
            self.trajectories[name] = []
            self.carrying[name] = []
            while len(self.trajectories[name]) < self.num_moves:
                c = random.choice(places)
                if c not in self.trajectories[name]:
                    self.trajectories[name].append(c)
                    if c not in self.places:
                        self.places.append(c)
        self.object_locations = {}
        for o in self.objects:
            c = random.choice(self.places)
            self.object_locations[o] = c
        done = False
        last_name = None
        while done != True:
            name = random.choice(self.names)
            while name == last_name:
                name = random.choice(self.names)
            last_name = name
            if name not in self.people_seen:
                self.people_seen.append(name)
            location = self.trajectories[name][self.people_locations[name]]
            if self.people_locations[name] == 0:
                s = [name, "is", "in", "the", location]
                if s not in self.sent:
                    self.sent.append([name, "is", "in", "the", location])
            obs = []
            for o, l in self.object_locations.items():
                if l == location:
                    obs.append(o)
                    if random.random() > 0.4:
                        if o not in self.objects_seen:
                            self.objects_seen.append(o)
                        if random.random() > 0.5:
                            self.sent.append([name, random.choice(observe_actions), "a", o, "there"])
            peeps = []
            for p, l in self.people_locations.items():
                if l == location:
                    if p != name:
                        peeps.append(p)
                        if random.random() > 0.4:
                            if p not in self.people_seen:
                                self.people_seen.append(p)
                            if random.random() > 0.5:
                                self.sent.append([name, random.choice(observe_actions), p, "there"])
            if len(self.carrying[name]) > 0:
                if random.random() > 0.5:
                    o = random.choice(self.carrying[name])
                    if o not in self.objects_seen:
                        self.objects_seen.append(o)
                    self.carrying[name].remove(o)
                    self.object_locations[o] = location
                    act = random.choice(put_down_actions)
                    s = []
                    s.append(name)
                    for a in act:
                        if a != "OBJ":
                            s.append(a)
                        else:
                            s.append(o)
                    self.sent.append(s)
            if len(obs) > 0:
                if random.random() > 0.5:
                    o = random.choice(obs)
                    if o not in self.objects_seen:
                        self.objects_seen.append(o)
                    self.carrying[name].append(o)
                    self.object_locations[o] = ""
                    act = random.choice(pick_up_actions)
                    s = []
                    s.append(name)
                    for a in act:
                        if a != "OBJ":
                            s.append(a)
                        else:
                            s.append(o)
                    self.sent.append(s)
            if self.people_locations[name] < self.num_moves - 1:
                if random.random() > 0.5:
                    loc = self.people_locations[name]
                    loc += 1
                    next_location = self.trajectories[name][loc]
                    loc = self.people_locations[name] = loc
                    s = [name, random.choice(to_the_actions), "to", "the", next_location]
                    self.sent.append(s)
            ls = self.people_locations.values()
            if min(ls) == self.num_moves - 1:
                done = True
        self.set_question()

    def get_carrying(self):
        c = {}
        for n in self.names:
            if len(self.carrying[n]) > 0:
                for o in self.carrying[n]:
                    c[o] = n
        return c

    def set_question(self):
        questions = [self.object_question,
                     self.carry_question,
                     self.name_question,
                     self.location_question]
        q = []
        a = ""
        ret1 = None
        ret2 = None
        while ret1 == None:
            c = random.choice(questions)
            ret1, ret2 = c()
        q = ret1
        a = ret2
        q.append("?")
        self.question = q
        self.sent.append(q)
        self.answer = a

    def carry_question(self):
        car = self.get_carrying()
        if len(car.items()) < 1:
            return None, None
        q = ["who", "has", "the"]
        a = ""
        c = self.get_carrying()
        if len(c.items()) > 0:
            item = random.choice(list(c.keys()))
            holder = c[item]
            q.append(item)
            a = holder
        return q, a

    def object_question(self):
        if len(self.objects_seen) < 1:
            return None, None
        q = ["where", "is", "the"]
        a = ""
        o = random.choice(self.objects_seen)
        q.append(o)
        if self.object_locations[o] != "":
            a = self.object_locations[o]
        else:
            p = ""
            for n in self.names:
                if o in self.carrying[n]:
                    loc = self.trajectories[n][self.people_locations[n]]
                    a = loc
                    break
        return q, a

    def name_question(self):
        if len(self.people_seen) < 1:
            return None, None
        q = ["where", "is"]
        a = ""
        p = random.choice(self.people_seen)
        q.append(p)
        loc = self.trajectories[p][self.people_locations[p]]
        a = loc
        return q, a

    def location_question(self):
        q = ["who", "is", "in", "the"]
        a = ""
        locs = {}
        l_count = Counter()
        for name in self.names:
            loc = self.people_locations[name]
            location = self.trajectories[name][loc]
            locs[location] = name
            l_count[location] += 1
        chosen = ""
        tries = 0
        while chosen == "":
            if tries > 15:
                return None, None
            tries += 1
            l = list(locs.keys())
            r = random.choice(l)
            if l_count[r] == 1:
                chosen = r
                a = locs[r]
        q.append(chosen)
        return q, a




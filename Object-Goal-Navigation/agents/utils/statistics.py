
class Stat_Tool():
    def __init__(self,n_obj):
        self.n_obj = n_obj
        self.episode_no_by_objective = [0 for i in range(n_obj)]
        self.success_score_by_objective = [
            [] for i in range(n_obj)
        ]
        self.false_success_score_by_objective = [
            [] for i in range(n_obj)
        ]

        self.eval_stats_by_objective = [{
            'total_number':0.0,
            'success':0.0,
            'item_found':0.0
        } for i in range(n_obj)]

        self.eval_stats = {
            'total_number': 0.0,
            'success': 0.0,
            'item_found': 0.0
        }

        self.scenes = {}

    def update(self,obj_id,suc,goal_found):
        self.eval_stats['total_number'] += 1.0
        self.eval_stats_by_objective[obj_id]['total_number'] += 1.0
        if suc:
            self.eval_stats['success'] += 1.0
            self.eval_stats_by_objective[obj_id]['success'] += 1.0
        if goal_found:
            self.eval_stats['item_found'] += 1.0
            self.eval_stats_by_objective[obj_id]['item_found'] += 1.0

    def update_scene(self,scene):
        if scene not in self.scenes.keys():
            self.scenes[scene] = 0.0
        self.scenes[scene] += 1.0


    def show_stats(self):
        print(" ")
        print(" ")
        tn = self.eval_stats['total_number']
        print('total episode', tn)
        for m, v in self.eval_stats.items():
            print(m, v / tn, end=' ')
        print(' ')
        print('by object')
        for i in range(self.n_obj):
            if self.eval_stats_by_objective[i]['total_number'] < 1.0:
                continue
            obj_tn = self.eval_stats_by_objective[i]['total_number']
            print(i, 'episode no', obj_tn)
            for m, v in self.eval_stats_by_objective[i].items():
                print(m, v / obj_tn, end=' ')
            print(' ')
        print("scenes",self.scenes)


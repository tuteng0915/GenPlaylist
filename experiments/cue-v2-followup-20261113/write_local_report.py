"""Format verified Stage A follow-up evidence. No paper repo changes."""
import json,hashlib
from pathlib import Path
ROOT=Path('/Users/teng/Documents/Codex/2026-11-04/viddual-session-4/outputs/GenPlaylist-cue-v2-followup-20261113')
D=ROOT/'results-20261113T0121'
def read(n):return json.loads((D/n).read_text())
def pair(v):return f"{v['delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
def table(headers,rows):return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,r)) for r in rows])
lines=['Stage A 收尾核查报告（2026-11-13 01:20 SGT）','',
'本报告的 v2 均指原始高空槽版本；修正版单独标为 r2。未进入 Stage B，未修改论文、未 push、未转发其他 session。',
'原四组每个数据集200上下文，合计1600段音频，MMR新增400段已完成。哈希、同一history/target/SID/seed、原解码验证覆盖均复核通过。原队列152/156任务完成、0失败；未完4项是真实性审核、40首人工审核材料导出、依赖审核的最终总验证与打包。','',
'一、v1 相对无 cues：配对差值与95%区间','差值=v1−无cues；MERT-OAS和历史拟合越高越好，FAD越低越好。200上下文为重采样单位；标量10000次，FAD1000次，同一重采样索引用于各组和真实音频。FAD为每次重采样重新计算的分布距离，不是伪造的逐样本FAD。MERT-OAS是本次单音频对5真实目标的1×5匹配，不是5×5播放列表评测。']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 p=read(ds+'/paired-gate.json')['pairs']['v1 minus none']
 rows.append([label]+[pair(p[k]) for k in ['mert_oas','history_fit','fad']])
lines+=[table(['数据集','MERT-OAS Δ [95% CI]','历史拟合 Δ [95% CI]','FAD Δ [95% CI]'],rows),
'判断：两数据集的MERT-OAS/历史拟合均无明确提升；MPD三项区间均跨0。Music4All的FAD明显变差。不能据此声称第一首真实目标歌曲的v1 cues改善了整体音频生成。','',
'二、全目录集中度（均为每首8个cue）']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 q=read(ds+'/catalog-concentration.json')
 for a in ['v1','mmr']:
  v=q[a];rows.append([label,a,v['songs'],f"{v['max_song_fraction']:.2%}",f"{v['top10_occurrence_share']:.2%}",f"{v['entropy_bits']:.4f}",f"{v['pop_rap_song_fraction']:.2%}"])
lines+=[table(['数据集','版本','目录歌曲','最高单cue歌曲占比','前10占全部cue出现次数','熵(bits)','pop rap歌曲占比'],rows),
'口径：每首8个不重复cue；前10占比的分母为8×歌曲数。原16槽数据仅在JSON中作背景，未混入本表。两个目录的v1 Top8均完整重现（MPD5119/5119、Music4All10950/10950），集中度由独立计数程序复算通过。',
'MMR通过歌曲内部去冗余间接缓解了跨歌曲集中；但它没有全目录频率约束，Music4All仍有42.93%的歌曲携带pop rap。','',
'三、固定真实参考的内容指标','同一个all-mpnet-base-v2编码器，300token分块并编码全部文本。歌词指标：生成的lyric_draft对5首真实目标歌曲的已有歌词；属性指标：生成music_attributes对目标genre/tags/mood，排除cue词表、歌名和artist。这里评估送入渲染器的文本，不是从合成音频转写歌词或检测属性。',
'MPD有效歌词999/1000目标出现次数，覆盖200上下文；Music4All913/1000，覆盖198上下文。所有组使用相同缺失掩码；元数据两数据集均1000/1000。Music4All的853首唯一参考歌中628份为词干化/tokenized歌词，155份为可用行式歌词，70份缺失/过短；该指标不能消除源歌词质量问题。']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 q=read(ds+'/independent-content-complete.json')
 for a in ['v1','mmr','v2','none']:
  v=q['arms'][a];rows.append([label,a,f"{v['actual_lyric_fit']['mean']:.5f}",f"{v['actual_metadata_fit']['mean']:.5f}"])
lines+=[table(['数据集','版本','真实歌词相似度','真实元数据相似度'],rows),'配对差值（10000次bootstrap，全部6对见JSON）：']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 q=read(ds+'/independent-content-complete.json')
 for p in ['v1 minus none','mmr minus none','mmr minus v1','v2 minus none']:
  v=q['paired'][p];rows.append([label,p,pair(v['actual_lyric_fit']),pair(v['actual_metadata_fit'])])
lines+=[table(['数据集','配对','歌词 Δ [95% CI]','元数据 Δ [95% CI]'],rows),
'判断：MPD的v1/MMR提高了歌词文本贴合，但元数据贴合下降；MMR相对v1的差异区间跨0。Music4All的v1/MMR相对无cues的两项区间均跨0。原v2接近无cues，不能代表完整逐歌cue构造的效果。以下多个区间为探索性、未做多重比较校正。','',
'四、保留原Content fit，并列报告四组完整均值']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 q=read(ds+'/original-gate-report.json')
 for a,v in q['arms'].items():rows.append([label,a]+[f"{v[k]['mean']:.5f}" for k in ['mert_oas','history_fit','content_fit_v1','content_fit_v2']]+[f"{v['fad']:.5f}"])
lines+=[table(['数据集','版本','MERT-OAS','历史拟合','Content fit(v1参考)','Content fit(v2参考)','FAD'],rows),
'各组均值区间保存在original-gate-report.json；全部6对、包括两个旧Content fit参考的差值区间保存在all-gate-pairs.json。MMR−无cues的MERT-OAS和历史拟合区间均跨0；Music4All FAD Δ=+0.80167 [0.24284,1.38198]。MMR改善cue分布不等于已改善音频指标。','',
'五、v2故障定位与r2诊断','原实现把共享歌曲数<5的簇直接丢弃，4987簇只留下95簇，丢弃4892簇，确实偏离“合并到更概括概念”的要求。另一项medoid映射半径风险在合成测试中成立，但对这批保留下来的候选实际影响为0，不把它误报为本次主要原因。仅映射阈值0.78→0.65，两个试点非空槽也只从62→104、33→64。',
'r2改用全部3544首候选，支持数下限2，加入同类别有界语义合并，并按自身簇成员关系映射。原候选和全部未解决候选保留，原v1/v2未覆盖。r2是诊断性修复，尚不是合格发布版：部分低共享概念没有安全父簇，另有77个簇超出2047真实ID预算；“向上合并”的语义正确性也尚未人工核验，不能把余弦接近等同于概念上位关系。']
f=read('repair/diagnostic-funnel.json');checks=read('repair/independent-checks.json');rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 d=f['datasets'][ds];c=d['funnel'];g=d['coverage']['all'];rows.append([label,c['songs'],c['raw_candidates'],c['after_evidence_filter'],checks[ds]['canonical_clusters_with_dataset_support'],c['mapped_candidates'],c['selected_slots'],f"{g['coverage_on_lyric_songs']:.2%}",f"{g['candidate_count_upper_bound_coverage']:.2%}"])
lines+=[table(['数据集','歌曲','候选','证据过滤后','规范cue种类*','映射候选','最终槽位','有歌词槽位覆盖','现有候选理论上限'],rows),
'*规范cue种类与候选出现次数不是同一单位；共享词表共2047真实ID，两数据集可以使用相同ID。覆盖分母：MPD1659有歌词歌曲×8，Music4All1544×8。',
'主要剩余问题已定位到提取阶段：原始未过滤候选本身平均每歌仅6.74/4.48个；即使取消全部过滤，其有歌词歌曲的槽位上限也仅84.89%/64.55%，仍达不到90%。不能靠降低阈值、复用同义词填满或放入无证据候选解决。需要重新设计候选提取与证据定位，再验证语义合并。','同一100首试点的诊断：']
rows=[]
for ds,label in [('mpd','MPD'),('m4a','Music4All')]:
 q=read('repair/diagnostics.json')['datasets'][ds]
 for a,v in q['arms'].items():rows.append([label,a,f"{v['nonempty_slots']}/800",f"{v['mean_pairwise_cosine']:.4f}",f"{v['retrieval_mrr']:.4f}",f"{v['max_song_fraction']:.2%}",f"{v['top10_occurrence_share']:.2%}",f"{v['entropy_bits']:.4f}"])
lines+=[table(['数据集','版本','非空槽','歌内平均相似度↓','相似歌曲MRR↑','最高单cue占歌比','前10占比','熵'],rows),
'MRR是同曲风/近情绪候选池的检索诊断，MPD45首、Music4All53首符合候选数≥5条件；不是推荐Recall，也不是独立留出泛化检验。冗余度只对至少2个有效cue的歌计算，v2/r2的低覆盖会影响比较。r2有歌词试点的覆盖为40.00%/23.73%，全100首分母下则38.00%/18.75%。',
'90%门槛未通过，因此修正版400段音频没有启动；原v2音频不能作为修正版结果。','',
'六、完成情况、耗时和预计时间','已完成：原四组1600音频及生成指标/区间、新两项固定真实参考指标/6对区间、v1/MMR全目录集中度、r2映射诊断与独立复核。',
'仍在运行：原计划的盲评真实性审核240/600个“歌曲×版本”记录。重启后约68分钟完成64条，剩余360条按当前速度估计还需6–8小时，预计11/13 07:30–09:30 SGT；完成后40首人工审核材料与最终打包仍需约10–20分钟。该估计不是完成承诺；现有任务受11:11:30的预约截止保护，最晚11:13前释放。尚不能给出完整真实性提升结论，也没有代替作者填写人工审核。',
'修正版尚未完成：候选重提取、语义上位合并、90%覆盖达标及其后400音频。未启动新一轮提取或修正版音频，暂不能给出可靠完工时间；必须先由小样本证明确实能达到覆盖与真实性要求。',
'本次新增文本计算的实际GPU步骤：两数据集独立内容指标81.58秒，修正版相同试点诊断49.59秒，接续Music4All全目录MMR110.07秒；MPD目录先前CPU已完成。映射定位CPU93.25秒、r2构造CPU62.70秒。此前中止的低优先级CPU尝试另在cpu-to-free-gpu-transition.json记录，不与GPU完成耗时混同。',
'原Stage A已完成任务的耗时如下（worker时间之和不是并行墙钟时间）：']
rows=[]
for k,v in read('queue-status-and-timing.json')['timing'].items():rows.append([k,v['completed_tasks'],f"{v['sum_worker_seconds']/60:.2f}",f"{(v['last_end_epoch']-v['first_start_epoch'])/60:.2f}" if v['first_start_epoch'] else 'N/A'])
lines+=[table(['任务组','完成任务数','worker分钟合计','首开始至末结束分钟*'],rows),
'*首尾时间包含穿插等待，任务组互有重叠，不能相加作为总墙钟时间。',
'本次汇报后停止新增实验，等待作者决定cue版本和是否进入Stage B。已在运行的原审核及其依赖验证按原有限计划完成；不创建周期任务、不自动同步、不push。','',
'证据文件：同目录results-20261113T0121，含数值JSON、全部配对、验证报告和源SHA256。大批歌词逐文件hash仍保留在远端原报告，本机精简副本明确记录省略字段、源位置和完整源文件hash。']
(ROOT/'REPORT-20261113.txt').write_text('\n\n'.join(lines)+'\n')
(D/'LOCAL-SHA256.json').write_text(json.dumps({str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*.json') if p.name!='LOCAL-SHA256.json'},indent=2)+'\n')
print(ROOT/'REPORT-20261113.txt')

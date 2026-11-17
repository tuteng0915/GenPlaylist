"""Prepare a local-to-cluster Stage A result pack; no push or paper changes."""
import shutil
from common import *

def main():
 verification=json.loads((O/'verification.json').read_text());assert verification['passed']
 dest=O/'delivery';dest.mkdir(exist_ok=True);files=['verification.json','diagnostics.json','faithfulness.json','human-audit-40.html','build-report.json','extraction-config.json','judge-config.json','text-encoder-assets.json','audio-source-audit.json','preflight.json','offensive-categorized-scan.json','offensive-generated-text-scan.json','v2/cue_vocab.json','v2/cue_categories.json','v2/vocabulary-support.json']
 for ds in ['mpd','m4a']:files.extend(f'{ds}/{f}' for f in ['cohort.json','gate-report.json','gate-per-context.npz','content-config.json','v1-reconstruction-check.json','v1-mmr-item2cues.json','v1-mmr-gate-item2cues.json','v1-gate-reconstruction-check.json','v2-item2cues.json','v2-selection-evidence.json'])
 for rel in files:
  p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/rel,p)
 config=dict(stage='A only',stage_B='not authorized; no full rebuild or training',personalization='full v2 diagnostic deferred to Stage B by author',scripts={p.name:sha(p) for p in E.glob('*.py')},artifacts={rel:sha(dest/rel) for rel in files},successful_tasks={p.stem:json.loads(p.read_text()) for p in (O/'done').glob('*.json')},process_records={p.name:json.loads(p.read_text()) for p in (O/'processes').glob('*.json')},old_attempts='pilot-debug contains extraction prompt/evidence repair and output-cap repair records; completed valid results were retained',paper_changed=False,push=False)
 atomic(dest/'run_config.json',config)
 diag=json.loads((O/'diagnostics.json').read_text())['datasets'];faith=json.loads((O/'faithfulness.json').read_text());scan=json.loads((O/'offensive-categorized-scan.json').read_text());audio=json.loads((O/'audio-source-audit.json').read_text())
 lines=['# Cue v2：Stage A 结果包','', '本轮仅包含每数据集100首试点和200上下文×4条件生成测试，共1600段新音频（按作者后续指示增加MMR条件）。v2全量个性化诊断留到Stage B。没有启动全目录重建或模型训练。', '', '## 阅读口径','', 'MERT-OAS为1段生成对5首目标的矩形最优匹配，等价于取最大相似度，不是此前5×5整列表指标。同时报告目标均值和首曲相似度。FAD参考200个首曲目标出现次数；Content fit对5首目标取均值，分别使用v1/v2参考cue。所有区间按上下文成对重采样。', '', '真实性由同系列Qwen模型用独立、隐藏组名的提示进行判断，不等同于独立人工结论。请结合human-audit-40.html检查；留空槽位、非空槽支持率和全部8槽支持率同时报告。', '']
 for ds in ['mpd','m4a']:
  lines.extend(['## '+ds.upper(),'','| 条件 | 自动支持率/8槽 | 自动支持率/非空槽 | NONE槽 | 前10词占比 | 歌内平均余弦 | 近似歌MRR |','|---|---:|---:|---:|---:|---:|---:|'])
  for arm in ['v1','mmr','v2']:
   d=diag[ds]['arms'][arm];f=faith[ds][arm];lines.append(f"| {arm} | {f['support_rate_all_slots']:.4f} | {f['support_rate_nonempty']:.4f} | {d['none_slots']}/800 | {d['top10_occurrence_share']:.4f} | {d['mean_pairwise_cosine']} | {d['retrieval_mrr']} (n={d['retrieval_n']}) |")
  r=json.loads((O/ds/'gate-report.json').read_text());lines.extend(['','| 渲染条件 | MERT-OAS ↑ | Content fit/v1 ↑ | Content fit/v2 ↑ | History fit ↑ | FAD ↓ |','|---|---:|---:|---:|---:|---:|'])
  for arm in ['v1','mmr','v2','none']:
   d=r['arms'][arm];lines.append(f"| {arm} | {d['mert_oas']['mean']:.5f} | {d['content_fit_v1']['mean']:.5f} | {d['content_fit_v2']['mean']:.5f} | {d['history_fit']['mean']:.5f} | {d['fad']:.5f} |")
  lines.extend(['','配对差值及95%区间见'+ds+'/gate-report.json；正负方向如实保留，不自动宣告通过。',''])
 racial=sum(g['category_hit_records']['racial_slur'] for g in scan['groups'].values());other=sum(g['category_hit_records']['other_profanity'] for g in scan['groups'].values());anyhit=sum(g['any_screen_hit_records'] for g in scan['groups'].values())
 lines.extend(['## 数据核查','',f"原有{scan['records']}条生成记录中，{anyhit}条命中词面筛查；其中种族歧视性词形命中{racial}条，其他粗口命中{other}条，可重叠。记录数不等于去重后的歌词数；样例ID与路径在JSON中，未删除任何历史产物。",'',f"MPD音频：{audio['datasets']['mpd']['passed']}/{audio['datasets']['mpd']['checked']}通过文件头、时长和首秒解码检查。Music4All：测试需要的{audio['datasets']['m4a']['passed']}个不同首曲音频通过检查，覆盖200次目标出现。",'', '乐器关键词命中不等于经独立验证的乐器标签。当前两个数据集均采用genre/tags作为sound的文字证据，未把歌词推断当作音频特征。歌词质量检查可识别短文本和部分列表污染，不能保证发现所有错歌。', '', '## 耗时与复现','', 'run_config.json记录各成功任务的实际耗时、命令、脚本哈希与进程。pilot-debug保留前期修正记录。所有完成音频通过哈希、种子、history、cue、时长检查；MERT-OAS与history fit另行复算。', '', '**Stage A 到此结束。是否进入Stage B由作者决定。**'])
 (dest/'READ-ME.md').write_text('\n'.join(lines)+'\n')
 atomic(O/'stage-A-complete.json',dict(completed_epoch=time.time(),delivery=str(dest),report_sha256=sha(dest/'READ-ME.md'),stage_B_started=False))

if __name__=='__main__':main()

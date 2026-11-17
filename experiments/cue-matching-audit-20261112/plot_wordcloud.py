"""Data-derived phrase clouds with presentation-only offensive-label filtering."""
import argparse,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

OFFENSIVE_LABEL = re.compile(r"\b(?:nigg(?:a|as|er|ers)|fagg?(?:ot|ots)|cunts?)\b", re.I)

def display_allowed(phrase):
    return not OFFENSIVE_LABEL.search(phrase)

def main(root):
    out=Path(root)/'outputs/GenPlaylist-cue-matching-audit-20261112';data=json.loads((out/'cue-frequencies.json').read_text());figdir=out/'figures';figdir.mkdir(exist_ok=True)
    (figdir/'presentation-policy.json').write_text(json.dumps({'scope':'Visualization only; source cues and evaluation unchanged','wordcloud':'Omit offensive labels before selecting top 80','frequency_chart':'Redact offensive labels, preserve original ranks and counts','omitted_counts':{ds:sum(not display_allowed(r['phrase']) for r in data[ds]['catalog']) for ds in ['mpd','m4a']}},indent=2))
    font=font_manager.findfont('DejaVu Sans')
    palette=['#92520F','#AB631B','#BF792C','#D48B34','#9C682F','#704C24']
    def color(word,font_size,position,orientation,random_state=None,**kw):return palette[random_state.randint(0,len(palette)-1)]
    fig,axs=plt.subplots(1,2,figsize=(16,7),dpi=180,facecolor='white')
    for ax,ds,title in zip(axs,['mpd','m4a'],['MPD','Music4All-Onion']):
        d=data[ds];freq={r['phrase']:r['count'] for r in d['catalog']}
        assert len(freq)==len(d['catalog']),'Duplicate vocabulary strings require explicit treatment'
        freq={phrase:count for phrase,count in freq.items() if display_allowed(phrase)}
        wc=WordCloud(width=1300,height=790,background_color='white',font_path=font,max_words=80,max_font_size=160,min_font_size=13,prefer_horizontal=1.0,relative_scaling=.5,random_state=42,collocations=False,margin=5,color_func=color).generate_from_frequencies(freq)
        wc.to_file(str(figdir/f'cue-wordcloud-{ds}.png'));(figdir/f'cue-wordcloud-{ds}.svg').write_text(wc.to_svg(embed_font=False))
        ax.imshow(wc,interpolation='bilinear');ax.axis('off');ax.set_title(title,loc='left',fontsize=19,fontweight='bold',pad=35,color='#22303F')
        ax.text(0,1.025,f"{d['catalog_size']:,} songs · 8 catalog cues per song",transform=ax.transAxes,fontsize=11,color='#566371')
        ax.text(0,-.07,f"Top 10 cues: {100*d['top10_occurrence_share']:.1f}% of all cue occurrences",transform=ax.transAxes,fontsize=12,color='#704C24')
    fig.suptitle('Catalog cue frequency',x=.04,y=.97,ha='left',fontsize=25,fontweight='bold',color='#22303F')
    fig.text(.04,.905,'Top 80 phrases per dataset · offensive labels omitted · word sizes scaled within each panel',fontsize=11,color='#566371')
    fig.text(.04,.035,'Counts describe automatically assigned catalog cues, not independently verified musical genres.',fontsize=10,color='#566371')
    fig.subplots_adjust(left=.04,right=.98,top=.81,bottom=.13,wspace=.10)
    fig.savefig(figdir/'cue-wordcloud-two-datasets.png',dpi=180,facecolor='white');plt.close(fig)
    # A quantitative companion makes the frequency skew readable without relying on word area.
    fig,axs=plt.subplots(1,2,figsize=(13,5.8),dpi=180)
    for ax,ds,title in zip(axs,['mpd','m4a'],['MPD','Music4All-Onion']):
        rows=data[ds]['catalog'][:10][::-1];ys=list(range(10));vals=[100*r['song_fraction'] for r in rows]
        ax.barh(ys,vals,color='#D18A3B',height=.68);ax.set_yticks(ys,[r['phrase'] if display_allowed(r['phrase']) else '[label omitted]' for r in rows],fontsize=10);ax.set_xlim(0,85);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Catalog songs assigned this cue (%)');ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        for y,v in zip(ys,vals):ax.text(v+1,y,f'{v:.1f}',va='center',fontsize=9)
    fig.tight_layout(w_pad=3);fig.savefig(figdir/'cue-frequency-top10.png');fig.savefig(figdir/'cue-frequency-top10.svg');plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);main(p.parse_args().workspace)

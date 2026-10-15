'use strict';
const $ = id => document.getElementById(id);
const modeNames = {idle:'仅监控',free:'松弛 · 手动示教',hold:'已锁定当前位置',playback:'动作回放中',disabled:'输出已停止',fault:'故障 / 急停锁存'};
const jointLabels = ['髋 Pitch','髋 Roll','髋 Yaw','膝 Pitch','踝 Pitch','踝 Roll'];
let token='', state=null, points=[], dirty=false, busy=false, lastFileSignature='', fileCache=[];
function notice(message, error=false) { $('notice').textContent=message; $('notice').classList.toggle('error',error); }
async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json','X-Teaching-Token':token},body:JSON.stringify(body),signal:AbortSignal.timeout(5000)} : {cache:'no-store',signal:AbortSignal.timeout(3000)});
  const data=await response.json();
  if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function command(action, extra={}) {
  if (busy && !['estop','disable'].includes(action)) { notice('上一条请求仍在处理中，请稍候再点击。',true); return false; }
  busy=true;
  try {
    const result=await api('/api/command',{action,...extra});
    render(result);
    if (['checkpoint','new-checkpoints','load-checkpoints','save-checkpoints'].includes(action)) {
      points=structuredClone(result.checkpoints); dirty=false; renderPoints();
    }
    notice(`${({release:'已请求 Kp/Kd=0；查看下方 CAN 提交确认状态',hold:'已请求锁定最新实测角度；等待 STM32 CAN 提交确认',disable:'停止输出请求已发送',estop:'急停请求已发送',reset:'故障锁存已清除；仍保持停止输出',reconnect:'USB 已重新打开，当前仅监控；需主动点击松弛','start-recording':'已开始记录实际反馈帧','stop-recording':'记录已停止并保存',checkpoint:'当前姿态已保存','save-checkpoints':'区间时间与顺序已保存','play-log':'连续动作开始回放','play-checkpoints':'Checkpoint 动作开始回放','stop-playback':'回放已停止，已请求锁定当前姿态'})[action] || '操作完成'}`);
    await refreshFiles();
    return true;
  } catch(error) { notice(error.message,true); return false; }
  finally {busy=false;}
}
function element(tag,cls,text) {const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
for(let row=0;row<6;row++) for(let leg=0;leg<2;leg++) {
  const i=leg*6+row, e=element('div','joint');
  const name=element('span','name');name.append(element('span','index',String(i).padStart(2,'0')),document.createTextNode(`${leg?'右':'左'} ${jointLabels[row]}`));
  const angle=element('span','angle','—');angle.id=`angle${i}`;
  const target=element('small','','目标 —');target.id=`target${i}`;
  const freshness=element('small','','CAN 年龄：未知');freshness.id=`canAge${i}`;
  e.append(name,angle,target,freshness);$('joints').append(e);
}
const diagPanel=element('div','can-panel');diagPanel.id='canPanel';
$('joints').after(diagPanel);
const reconnectButton=element('button','quiet','重新连接 USB');
reconnectButton.addEventListener('click',()=>command('reconnect'));
document.querySelector('.controls').append(reconnectButton);
function render(s) {
  state={...state,...s}; s=state;
  $('mode').textContent=modeNames[s.mode] || s.mode;
  if(['free','hold','playback'].includes(s.mode) && s.gain_confirmation!=='submitted') $('mode').textContent+=' · 尚未确认';
  const fresh=s.state_age_s!==null && s.state_age_s<0.1;
  $('led').classList.toggle('online',fresh);
  $('connection').textContent=fresh?'USB 帧在线':'USB 帧离线';
  $('feedback').textContent=s.simulated?'模拟数据 · 无硬件连接':s.enable_motors?'电机控制已授权':'监控模式 · 未授权使能';
  $('demo').hidden=!s.simulated;
  $('age').textContent=s.state_age_s===null?'—':`${(s.state_age_s*1000).toFixed(0)} ms`;
  $('flags').textContent=`0x${s.status_flags.toString(16).toUpperCase().padStart(4,'0')}`;
  $('firmware').textContent=s.firmware_ready?(s.can_diagnostics?'CAN 诊断版':'示教旧版 · 需升级'):'需更新 STM32';
  const rad=$('unit').value==='rad', factor=rad?1:180/Math.PI, suffix=rad?' rad':'°';
  for(let i=0;i<12;i++) {
    $(`angle${i}`).textContent=s.position_rad?(s.position_rad[i]*factor).toFixed(rad?3:1)+suffix:'—';
    $(`target${i}`).textContent='目标 '+(s.target_rad[i]*factor).toFixed(rad?3:1)+suffix;
    const canAge=s.can_diagnostics ? s.can_diagnostics.feedback_age_ms[i]+(s.state_age_s||0)*1000 : null;
    $(`canAge${i}`).textContent=canAge===null?'CAN 年龄：旧固件未提供':`CAN ${canAge>=65535?'未收到':Math.round(canAge)+' ms'}${canAge>100?' · 过期':''}`;
    $(`canAge${i}`).classList.toggle('stale',canAge!==null&&canAge>100);
  }
  const d=s.can_diagnostics, confirmed=s.gain_confirmation==='submitted';
  diagPanel.classList.toggle('stale',!confirmed&&['free','hold','playback'].includes(s.mode));
  if(!d) diagPanel.textContent='缺少 CAN 诊断：当前 USB 在线不能证明编码器更新或松弛已生效。请烧录新版 STM32 固件。';
  else {
    const mode=['停止输出','Kp=Kd=0','恢复 PD'][d.applied_gain_mode]||'未知';
    diagPanel.textContent=`${s.gain_confirmation==='monitoring'?'仅监控，尚未发送控制指令':(confirmed?'STM32 已向 12 电机提交':'等待 STM32 提交')+'：'+mode} · 命令序号 ${d.applied_command_sequence} ｜ USB 序号 ${s.state_sequence} · 收命令 ${s.command_rx_count} · 控制周期 ${s.system_control_cycle}\n`+
      [1,0].map((b)=>`${b?'左腿 CAN2':'右腿 CAN1'}：TX失败 ${d.can_tx_failures[b]} · 恢复 ${d.can_recoveries[b]} · TEC ${d.can_ecr[b]&255} · REC ${(d.can_ecr[b]>>8)&127} · bus-off ${(d.can_psr[b]&128)?'是':'否'}`).join('\n');
  }
  $('gainHint').textContent=s.mode==='free'?'每 20 ms 回传最新反馈角度，Kp=Kd=0。':s.mode==='hold'?'目标固定；使用 config.py 原有站立 PD。':s.mode==='playback'?'所有关节按同一时间轴执行；完成后保持末帧。':'先托住机器人，再进入松弛模式。';
  $('recordBadge').textContent=s.recording?'● 记录中':'待机';$('recordBadge').classList.toggle('active',s.recording);
  $('samples').textContent=`${s.record_count} 帧`;$('missing').textContent=`缺帧估计：${s.record_missing}`;
  $('recordFile').textContent=s.record_name || '尚无记录';
  $('checkpointFile').textContent=s.checkpoint_name || '记录当前姿态时自动创建文件';
  $('startRecord').disabled=s.recording||s.mode!=='free';
  $('progress').textContent=s.playback_total_s?`${s.playback_elapsed_s.toFixed(1)} / ${s.playback_total_s.toFixed(1)} s`:'';
  if(s.error)notice(s.error,true);
  else if(!d)notice('当前固件不能确认逐关节 CAN 反馈和 PD 提交；锁定/记录/回放需新版固件。',true);
  if(!dirty && JSON.stringify(points)!==JSON.stringify(s.checkpoints)){points=structuredClone(s.checkpoints);renderPoints();}
}
function markDirty(){dirty=true;$('dirty').textContent='有未保存修改';}
function renderPoints(){
  const box=$('checkpoints');box.replaceChildren();$('dirty').textContent=dirty?'有未保存修改':'';
  if(!points.length){box.append(element('p','empty','记录第一个姿态，开始编排动作。'));return;}
  points.forEach((p,i)=>{
    const row=element('div','checkpoint');row.append(element('strong','small',`${i+1}`));
    const label=element('input');label.value=p.label;label.setAttribute('aria-label',`姿态 ${i+1} 名称`);label.addEventListener('input',()=>{p.label=label.value;markDirty();});row.append(label);
    const duration=element('div','duration');
    if(i<points.length-1){const input=element('input');input.type='number';input.min='0.02';input.step='0.1';input.value=p.duration_s;input.setAttribute('aria-label',`姿态 ${i+1} 到 ${i+2} 用时秒`);input.addEventListener('input',()=>{p.duration_s=Number(input.value);markDirty();});duration.append(input,element('span','small','s'));}
    else duration.append(element('span','last','末帧保持'));row.append(duration);
    const tools=element('div','tools');
    for(const [label,delta] of [['↑',-1],['↓',1],['×',0]]){const b=element('button','quiet',label);b.title=delta?`向${delta<0?'上':'下'}移动`:'删除姿态';b.disabled=delta&&(i+delta<0||i+delta>=points.length);b.addEventListener('click',()=>{if(delta)[points[i],points[i+delta]]=[points[i+delta],points[i]];else points.splice(i,1);markDirty();renderPoints();});tools.append(b);}
    row.append(tools,element('small','',p.position_rad.map(q=>(q*180/Math.PI).toFixed(1)+'°').join(' / ')));box.append(row);
  });
}
async function savePoints(){return await command('save-checkpoints',{name:state.checkpoint_name,checkpoints:points});}
async function refreshFiles(){
  const files=await api('/api/files');fileCache=files;
  const signature=JSON.stringify(files);if(signature===lastFileSignature)return;lastFileSignature=signature;
  const box=$('files');box.replaceChildren();
  if(!files.length){box.append(element('p','empty','还没有动作文件。完成记录后会显示在这里。'));return;}
  files.forEach(file=>{
    const checkpoints=file.name.endsWith('.json'),row=element('div','file'),info=element('div');
    info.append(element('div','file-name',file.name),element('div','file-info',`${checkpoints?'Checkpoint 姿态':'50 Hz 连续记录'} · ${(file.bytes/1024).toFixed(1)} KB`));
    const actions=element('div','row');
    if(checkpoints){const load=element('button','quiet','编辑');load.addEventListener('click',async()=>{if(dirty && !await savePoints())return;await command('load-checkpoints',{name:file.name});});actions.append(load);}
    const play=element('button','','回放');play.addEventListener('click',async()=>{
      if(checkpoints && file.name===state.checkpoint_name && dirty && !await savePoints())return;
      await command(checkpoints?'play-checkpoints':'play-log',{name:file.name,approach_s:Number($('approach').value),speed:Number($('speed').value)});
    });
    const download=element('a','','下载');download.href='/api/download?name='+encodeURIComponent(file.name);download.download=file.name;
    actions.append(play,download);row.append(info,actions);box.append(row);
  });
}
document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',async()=>{if(button.dataset.action==='new-checkpoints' && dirty && !await savePoints())return;await command(button.dataset.action);}));
$('startRecord').addEventListener('click',()=>command('start-recording'));
$('addCheckpoint').addEventListener('click',async()=>{if(dirty && !await savePoints())return;await command('checkpoint',{label:$('label').value || `姿态 ${points.length+1}`});$('label').value='';});
$('saveCheckpoints').addEventListener('click',savePoints);
$('refreshFiles').addEventListener('click',()=>refreshFiles().catch(e=>notice(e.message,true)));
$('unit').addEventListener('change',()=>state&&render(state));
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
async function poll(){
  try{render(await api('/api/state'));}
  catch(e){$('led').classList.remove('online');$('connection').textContent='服务连接中断';notice('无法连接示教服务。检查 SSH 隧道；服务器仍可能保持电机输出，请使用物理急停或重新连接。',true);}
  setTimeout(poll,100);
}
(async()=>{try{token=(await api('/api/session')).token;await refreshFiles();poll();}catch(e){notice(e.message,true);}})();

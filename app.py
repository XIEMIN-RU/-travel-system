import json
import uuid
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from flask import Flask, render_template, request, jsonify, session
import requests
import os
import time

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'taiwan-travel-dev-key')

# 暫存地圖 HTML 存成檔案，避免多 worker / 重啟後消失
MAP_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'map_cache')
os.makedirs(MAP_CACHE_DIR, exist_ok=True)

# ==========================================
# 0. 全域設定
# ==========================================
route_cache = {}
MAX_SEARCH_RADIUS = 20    
LUNCH_SEARCH_RADIUS = 15 
HOTEL_SEARCH_RADIUS = 15 

# 定義旅遊風格
STYLE_MAP = {
    " 不限風格 (隨機)": [],
    " 親子同樂": ["農場", "動物", "親子", "公園", "遊樂", "DIY", "觀光工廠", "休閒", "牧場"],
    " 大自然/戶外": ["步道", "山", "林", "溪", "瀑布", "生態", "風景", "自然", "花海", "森林", "濕地"],
    " 室內/文藝 (雨備)": ["博物館", "美術館", "展覽", "文物", "圖書館", "故事館", "室內", "文創", "教育"],
    " 網美/約會": ["景觀", "咖啡", "浪漫", "夜景", "藝術", "打卡", "老宅", "彩繪", "玻璃", "落羽松"],
    " 歷史/懷舊": ["古蹟", "廟", "老街", "文化", "遺址", "眷村", "歷史", "古厝"],
    " 商圈/購物": ["商圈", "購物", "百貨", "市集", "廣場", "outlet", "中心", "商店街", "商城"]
}

# ==========================================
# 1. 資料載入
# ==========================================
def load_data(file_path, key):
    if not os.path.exists(file_path): return pd.DataFrame()
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return pd.DataFrame(data.get(key, []))
    except: return pd.DataFrame()

df_attr = load_data('AttractionList.json', 'Attractions')
df_rest = load_data('RestaurantList.json', 'Restaurants')
df_hotel = load_data('HotelList.json', 'Hotels')

# ==========================================
# 2. 距離與路徑計算
# ==========================================
HEADERS = {'User-Agent': 'Mozilla/5.0'}   #模擬瀏覽器標頭，防止被 API 伺服器判定為惡意爬蟲而封鎖

def get_driving_distance(lat1, lon1, lat2, lon2):       #獲取行駛里程
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        res = requests.get(url, timeout=1.5, headers=HEADERS).json() 
        if res['code'] == 'Ok':
            return res['routes'][0]['distance'] / 1000          
    except: pass
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def get_real_route(lat1, lon1, lat2, lon2):              #獲取詳細路徑軌跡
    key = f"{lat1:.4f},{lon1:.4f}-{lat2:.4f},{lon2:.4f}"
    if key in route_cache: return route_cache[key]
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?geometries=geojson"
        res = requests.get(url, timeout=3, headers=HEADERS).json()
        if res['code'] == 'Ok':
            dist = res['routes'][0]['distance'] / 1000
            coords = [[p[1], p[0]] for p in res['routes'][0]['geometry']['coordinates']]
            route_cache[key] = (dist, coords)
            return dist, coords
    except: pass
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers, [[lat1, lon1], [lat2, lon2]]

# ==========================================
# 3. 篩選與搜尋邏輯
# ==========================================
def filter_by_city(df, city):  #地區與離島過濾
    if df.empty: return df
    city_data = df[df['PostalAddress'].apply(lambda x: x.get('City') == city if isinstance(x, dict) else False)].copy()
    city_data['_Town'] = city_data['PostalAddress'].apply(lambda x: x.get('Town'))
    offshore = ["琉球鄉", "綠島鄉", "蘭嶼鄉"]
    if city in ["屏東縣", "臺東縣"]:
        mainland = city_data[~city_data['_Town'].isin(offshore)]
        if not mainland.empty: return mainland.copy()
    return city_data

def apply_style_filter(df, keywords):  #關鍵字風格檢索
    if not keywords or df.empty: return df
    pattern = '|'.join(keywords)
    mask = df['AttractionName'].astype(str).str.contains(pattern, na=False) | \
           df['Description'].astype(str).str.contains(pattern, na=False)
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df

def find_strict_spot(lat, lon, target_df, priority_town=None, exclude_names=[], top_k=1, radius=MAX_SEARCH_RADIUS):
    if target_df.empty: return None        #排除重複>圓形初篩(畫半徑為MAX_SEARCH_RADIUS的圓)>行政區優先>精確排序
    candidates = target_df.copy()
    
    name_col = 'AttractionName' if 'AttractionName' in candidates.columns else \
               'RestaurantName' if 'RestaurantName' in candidates.columns else 'HotelName'
    
    if exclude_names:
        candidates = candidates[~candidates[name_col].isin(exclude_names)]
    
    if candidates.empty: return None

    # 初篩
    candidates['straight_dist'] = candidates.apply(
        lambda row: geodesic((lat, lon), (row['PositionLat'], row['PositionLon'])).kilometers, axis=1
    )
    potential_pool = candidates[candidates['straight_dist'] <= radius]
    
    if priority_town:
        town_matches = potential_pool[potential_pool['_Town'] == priority_town]
        if len(town_matches) >= 3: potential_pool = town_matches
    
    if potential_pool.empty: return None

    # 精確計算
    short_list = potential_pool.sort_values('straight_dist').head(20).copy()
    short_list['real_dist'] = short_list.apply(lambda row: get_driving_distance(lat, lon, row['PositionLat'], row['PositionLon']), axis=1)
    
    final = short_list.sort_values('real_dist').head(top_k)
    return final.sample(1).iloc[0] if not final.empty else None

def find_best_combo(current_lat, current_lon, curr_town, df_attrs, df_snacks, exclude_names=[]):
    """
    [修正] 同時傳入 exclude_names 給景點和點心，防止重複
    """
    if df_attrs.empty or df_snacks.empty: return None, None
    
    # 排除已去過的景點
    candidates = df_attrs[~df_attrs['AttractionName'].isin(exclude_names)].copy() if exclude_names else df_attrs.copy()
    if candidates.empty: return None, None

    candidates['straight_dist'] = candidates.apply(lambda row: geodesic((current_lat, current_lon), (row['PositionLat'], row['PositionLon'])).kilometers, axis=1)
    pool = candidates[candidates['straight_dist'] <= MAX_SEARCH_RADIUS]
    if pool.empty: return None, None

    top_attrs = pool.sort_values('straight_dist').head(5)
    best_combo = None
    min_score = float('inf')

    for _, attr_row in top_attrs.iterrows():
        a_lat, a_lon = attr_row['PositionLat'], attr_row['PositionLon']

        # 找點心時，也要排除已去過的店 (例如 D1 吃過的)
        nearest_snack = find_strict_spot(a_lat, a_lon, df_snacks, top_k=3, radius=5, exclude_names=exclude_names)
        
        if nearest_snack is not None:
            score = geodesic((current_lat, current_lon), (a_lat, a_lon)).kilometers + \
                    geodesic((a_lat, a_lon), (nearest_snack['PositionLat'], nearest_snack['PositionLon'])).kilometers
            if score < min_score:
                min_score = score
                best_combo = (attr_row, nearest_snack)
    
    if best_combo is None:
        return None, None
        
    return best_combo

# ==========================================
# 4. 單日行程規劃
# ==========================================
def plan_one_day(day_idx, start_pos, city_data_sets, num_spots, style_keywords, global_used_names):
    attrs_day, attrs_night, df_meal, df_snack, hotels = city_data_sets
    
    if style_keywords:
        attrs_pool = apply_style_filter(attrs_day, style_keywords)
    else:
        attrs_pool = attrs_day

    itinerary = []
    
    def get_title(suffix):
        return f"第{len(itinerary) + 1}站：{suffix}"

    curr_lat, curr_lon = start_pos['PositionLat'], start_pos['PositionLon']
    curr_town = start_pos.get('_Town')

    # --- 1. 決定當日第一站 ---
    if day_idx == 1:
        candidates = attrs_pool[~attrs_pool['AttractionName'].isin(global_used_names)]
        if candidates.empty: candidates = attrs_day
        
        start_node = candidates.sample(1).iloc[0].to_dict()
        start_node.update({'_type': 'attr', '_seq': get_title('出發')})
        itinerary.append(start_node)
        global_used_names.append(start_node['AttractionName'])
        
        curr_lat, curr_lon = start_node['PositionLat'], start_node['PositionLon']
        curr_town = start_node.get('_Town')
    else:
        morning_spot = find_strict_spot(curr_lat, curr_lon, attrs_pool, priority_town=curr_town, exclude_names=global_used_names, radius=10)
        
        if morning_spot is not None:
            ms_dict = morning_spot.to_dict()
            ms_dict.update({'_type': 'attr', '_seq': get_title('早晨散步')})
            itinerary.append(ms_dict)
            global_used_names.append(ms_dict['AttractionName'])
            
            curr_lat, curr_lon = ms_dict['PositionLat'], ms_dict['PositionLon']
            curr_town = ms_dict.get('_Town')

    # --- 2. 午餐 ---
    if not df_meal.empty:
        meal = find_strict_spot(curr_lat, curr_lon, df_meal, radius=LUNCH_SEARCH_RADIUS, exclude_names=global_used_names)
        if meal is not None:
            m_dict = meal.to_dict()
            m_dict.update({'_type': 'meal', '_seq': get_title('午餐')})
            itinerary.append(m_dict)
            global_used_names.append(m_dict['RestaurantName'])
            
            curr_lat, curr_lon = m_dict['PositionLat'], m_dict['PositionLon']
            curr_town = m_dict.get('_Town')

    # --- 3. 下午：景點+午茶 ---
    #  傳入 global_used_names 確保點心不重複
    best_attr, best_snack = find_best_combo(curr_lat, curr_lon, curr_town, attrs_pool, df_snack, global_used_names)
                                        #經緯度(上一站位置)、景點、城鎮、點心、已去過景點)
    if best_attr is not None:
        a_dict = best_attr.to_dict()
        a_dict.update({'_type': 'attr', '_seq': get_title('觀光')})
        itinerary.append(a_dict)
        global_used_names.append(a_dict['AttractionName'])
        curr_lat, curr_lon = a_dict['PositionLat'], a_dict['PositionLon']
        curr_town = a_dict.get('_Town')

        if best_snack is not None:
            s_dict = best_snack.to_dict()
            s_dict.update({'_type': 'snack', '_seq': get_title('午茶')})
            itinerary.append(s_dict)
            global_used_names.append(s_dict['RestaurantName']) # 記錄吃過的點心
            
            curr_lat, curr_lon = s_dict['PositionLat'], s_dict['PositionLon']
            curr_town = s_dict.get('_Town')
    else:
        fa = find_strict_spot(curr_lat, curr_lon, attrs_pool, priority_town=curr_town, exclude_names=global_used_names)
        if fa is not None:
            f_dict = fa.to_dict()
            f_dict.update({'_type': 'attr', '_seq': get_title('觀光')})
            itinerary.append(f_dict)
            curr_lat, curr_lon = f_dict['PositionLat'], f_dict['PositionLon']
            global_used_names.append(f_dict['AttractionName'])

    # --- 4. 補足剩餘景點 ---
    attr_count_in_day = len([x for x in itinerary if x['_type'] == 'attr'])
    
    while attr_count_in_day < num_spots:
        is_last = (attr_count_in_day == num_spots - 1)
        next_attr = None
        
        if is_last:
            nm = find_strict_spot(curr_lat, curr_lon, attrs_night, priority_town=curr_town, exclude_names=global_used_names)
            if nm is not None:
                next_attr, suffix = nm, "晚餐/夜市"
            else:
                next_attr, suffix = find_strict_spot(curr_lat, curr_lon, attrs_pool, exclude_names=global_used_names), "夜遊"
        else:
            next_attr, suffix = find_strict_spot(curr_lat, curr_lon, attrs_pool, exclude_names=global_used_names), "順遊"
        
        if next_attr is not None:
            na_dict = next_attr.to_dict()
            na_dict.update({'_type': 'attr', '_seq': get_title(suffix)})
            itinerary.append(na_dict)
            global_used_names.append(na_dict['AttractionName'])
            curr_lat, curr_lon = na_dict['PositionLat'], na_dict['PositionLon']
            attr_count_in_day += 1
        else:
            break
            
    # --- 5. 住宿 ---
    hotel_node = None
    if not hotels.empty:
        hotel = find_strict_spot(curr_lat, curr_lon, hotels, radius=HOTEL_SEARCH_RADIUS)
        if hotel is not None:
            hotel_node = hotel.to_dict()
            hotel_node.update({'_type': 'hotel', '_seq': get_title('住宿')})
            itinerary.append(hotel_node)
            
    return itinerary, hotel_node 

# ==========================================
# 5. 主程序：多天數地圖生成 (介面優化版)
# ==========================================
def generate_multi_day_map(city, total_days=2, num_spots_per_day=3, style_name=" 不限風格 (隨機)"):
    route_cache.clear()
    
    attrs = filter_by_city(df_attr, city)
    rests = filter_by_city(df_rest, city)
    hotels = filter_by_city(df_hotel, city)
    
    if len(attrs) < 5: return f"❌ {city} 資料不足，無法規劃多日遊。"

    is_night = attrs['AttractionName'].apply(lambda x: '夜市' in str(x))
    attrs_night = attrs[is_night].copy()    
    attrs_day = attrs[~is_night].copy()
    
    #  點心類別，避免被當成正餐
    snack_k = ['蛋糕', '甜點', '下午茶', '點心', '冰品', '咖啡', '烘焙', '豆花', '麻糬', '飲料', '茶', '冰淇淋', '刨冰', '雪花冰', '冰店', '甜湯', '粉圓', '仙草', '鬆餅', '舒芙蕾', '巧克力', '餅', '伴手禮', '菓子']
    
    is_snack = rests.apply(lambda x: (isinstance(x.get('CuisineClasses'), list) and 116 in x['CuisineClasses']) or 
                                     any(k in x.get('RestaurantName','') for k in snack_k), axis=1)
    
    df_snack = rests[is_snack]
    df_meal = rests[~is_snack & ~rests['RestaurantName'].astype(str).str.contains('夜市')]
    
    data_sets = (attrs_day, attrs_night, df_meal, df_snack, hotels)
    style_keywords = STYLE_MAP.get(style_name, [])

    js_tour_data = {} 
    last_night_hotel = {'PositionLat': attrs_day['PositionLat'].mean(), 'PositionLon': attrs_day['PositionLon'].mean(), '_Town': None}
    
    day_line_colors = ['blue', 'cadetblue', 'darkblue', 'darkpurple']
    global_used = [] 

    for day in range(1, total_days + 1):
        day_route, night_hotel = plan_one_day(day, last_night_hotel, data_sets, num_spots_per_day, style_keywords, global_used)
        if not day_route: continue
        
        if night_hotel: last_night_hotel = night_hotel
        else: last_night_hotel = day_route[-1]
        
        route_color = day_line_colors[(day-1) % len(day_line_colors)]
        
        day_spots = []
        day_route_points = []
        
        for item in day_route:
            lat, lon = item['PositionLat'], item['PositionLon']
            name = item.get('AttractionName') or item.get('RestaurantName') or item.get('HotelName')
            st = item['_type']
            
            img_url = ""
            if isinstance(item.get('Images'), list) and len(item['Images']) > 0:
                img_url = item['Images'][0].get('URL', '')
            
            desc = item.get('Description', '暫無介紹')
            if len(desc) > 80: desc = desc[:80] + "..."

            # 顏色配置
            if '夜市' in name:
                color = 'purple'; icon = 'star'
            elif st == 'meal':
                color = 'red'; icon = 'cutlery'
            elif st == 'snack':
                color = 'orange'; icon = 'cutlery'
            elif st == 'hotel':
                color = 'green'; icon = 'home'
            else:
                color = 'blue'; icon = 'camera'

            nav_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"

            # spot_type 供 Gemini prompt 使用
            if st == 'hotel':
                spot_type = '住宿'
            elif st in ('meal', 'snack'):
                spot_type = '餐廳'
            else:
                spot_type = '景點'

            day_spots.append({
                'lat': lat, 'lon': lon,
                'name': name,
                'title': item['_seq'],
                'nav': nav_url,
                'icon': icon,
                'color': color,
                'desc': desc,
                'img': img_url,
                'spot_id': f"d{day}s{len(day_spots)}",
                'spot_type': spot_type
            })
            
        for i in range(len(day_spots)-1):
            p1, p2 = day_spots[i], day_spots[i+1]
            dist, pts = get_real_route(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
            day_route_points.extend(pts)
            time.sleep(0.05)
            
        js_tour_data[day] = {
            'spots': day_spots,
            'route': day_route_points,
            'color': route_color 
        }

    start_pt = js_tour_data[1]['spots'][0]
    m = folium.Map(location=[start_pt['lat'], start_pt['lon']], zoom_start=11)
    folium.Marker([0,0], icon=folium.Icon(icon='info-sign', prefix='fa')).add_to(m)

    js_data_str = json.dumps(js_tour_data, ensure_ascii=False)
    map_var = m.get_name()

    day_btns_html = ""
    for d in range(1, total_days + 1):
        day_btns_html += f'<div class="day-btn" id="btn_day_{d}" onclick="switchDay({d})">D{d}</div>'

    day_options_html = ""
    for d in range(1, total_days + 1):
        day_options_html += f'<option value="{d}">第 {d} 天</option>'

    html_ui = f"""
    <div id="sidebar_days">{day_btns_html}</div>
    <div id="panel_container"></div>

    <div id="add-spot-panel">
        <div style="font-weight:bold; font-size:13px; margin-bottom:8px;">📍 新增自訂景點</div>
        <input id="custom-url" type="text" placeholder="貼上 Google Maps 連結"
            oninput="previewInsert()"
            style="width:100%; font-size:12px; padding:5px; margin-bottom:6px;
                   border:1px solid #ddd; border-radius:5px; box-sizing:border-box;">
        <input id="custom-name" type="text" placeholder="景點名稱（可自訂）"
            style="width:100%; font-size:12px; padding:5px; margin-bottom:6px;
                   border:1px solid #ddd; border-radius:5px; box-sizing:border-box;">
        <select id="custom-day" onchange="previewInsert()"
            style="width:100%; font-size:12px; padding:5px; margin-bottom:6px;
                   border:1px solid #ddd; border-radius:5px; box-sizing:border-box;">
            {day_options_html}
        </select>
        <div id="custom-preview"
            style="font-size:11px; color:#1976D2; min-height:14px; margin-bottom:6px;"></div>
        <button onclick="addCustomSpot()"
            style="width:100%; background:#2196F3; color:white; border:none;
                   padding:7px; border-radius:5px; cursor:pointer; font-size:12px;
                   font-weight:bold;">
            插入行程
        </button>
        <div id="custom-msg" style="font-size:11px; margin-top:5px; min-height:16px;"></div>
    </div>

    <style>
        #sidebar_days {{
            position: fixed; top: 120px; left: 10px; width: 50px;
            background: white; padding: 10px 5px;
            border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            z-index: 9000; display: flex; flex-direction: column; gap: 8px;
        }}
        #add-spot-panel {{
            position: fixed; top: 120px; left: 70px; width: 270px;
            background: white; padding: 12px 14px;
            border-radius: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            z-index: 9000; font-family: Arial;
            transition: opacity 0.2s;
        }}
        #add-spot-panel:hover {{ opacity: 1 !important; }}
        #add-spot-panel {{ opacity: 0.88; }}
        .day-btn {{
            width: 40px; height: 40px; line-height: 40px;
            background: #f0f0f0; border-radius: 50%;
            cursor: pointer; font-weight: bold; color: #555;
            text-align: center; font-family: Arial;
            transition: 0.2s; border: 2px solid #ddd;
        }}
        .day-btn:hover {{ background: #e0e0e0; }}
        .day-btn.active {{
            background: #2196F3; color: white; border-color: #1976D2; transform: scale(1.1);
        }}
        #panel_container {{
            position: fixed; bottom: 20px; left: 5%; width: 90%; height: 90px;
            z-index: 9000; display: flex; overflow-x: auto; 
            padding: 5px; gap: 10px; scrollbar-width: none;
        }}
        #panel_container::-webkit-scrollbar {{ display: none; }}
        .card {{
            min-width: 250px; height: 80px; background: white; 
            border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex; overflow: hidden; flex-shrink: 0;
            border-left: 6px solid #ccc; transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-3px); }}
        .card-img {{
            width: 80px; height: 100%; background-color: #eee;
            background-size: cover; background-position: center;
        }}
        .card-info {{ flex: 1; padding: 8px; cursor: pointer; display: flex; flex-direction: column; justify-content: center; }}
        .nav-btn {{
            width: 40px; background: #f8f9fa; color: #007bff;
            display: flex; align-items: center; justify-content: center;
            text-decoration: none; font-weight: bold; font-size: 14px;
            border-left: 1px solid #eee;
        }}
    </style>

    <script>
        var tourData = {js_data_str};
        var mapObject = null;
        var currentLayerGroup = null;
        var aiCity = '{city}';   // 由 Python 注入，供 Gemini prompt 使用

        window.onload = function() {{
            mapObject = {map_var};
            currentLayerGroup = L.layerGroup().addTo(mapObject);
            setTimeout(function() {{
                switchDay(1);
                loadAllAiDescs();   // 地圖就緒後開始背景載入 AI 導覽
            }}, 800);
        }};

        // ==========================================
        // 自訂景點：解析 Google Maps 連結
        // ==========================================
        function parseGoogleMapsUrl(url) {{
            var m;
            var num = '(-?[0-9]+\\.?[0-9]*)';
            // 格式1: ?q=lat,lon
            m = url.match(new RegExp('[?&]q=' + num + ',' + num));
            if (m) return {{ lat: parseFloat(m[1]), lon: parseFloat(m[2]) }};
            // 格式2: @lat,lon (一般地圖連結)
            m = url.match(new RegExp('@' + num + ',' + num));
            if (m) return {{ lat: parseFloat(m[1]), lon: parseFloat(m[2]) }};
            // 格式3: !3d lat !4d lon (分享連結)
            m = url.match(new RegExp('!3d' + num + '.*?!4d' + num));
            if (m) return {{ lat: parseFloat(m[1]), lon: parseFloat(m[2]) }};
            // 格式4: ll=lat,lon
            m = url.match(new RegExp('ll=' + num + ',' + num));
            if (m) return {{ lat: parseFloat(m[1]), lon: parseFloat(m[2]) }};
            return null;
        }}

        // 找最近站，回傳插入位置（住宿前）
        function findInsertIndex(dayNum, lat, lon) {{
            var spots = tourData[dayNum].spots;
            var bestIdx = 0, minDist = Infinity;
            for (var i = 0; i < spots.length; i++) {{
                if (spots[i].title && spots[i].title.includes('住宿')) continue;
                var d = Math.hypot(spots[i].lat - lat, spots[i].lon - lon);
                if (d < minDist) {{ minDist = d; bestIdx = i; }}
            }}
            // 插在最近站後面，但不超過住宿站
            var hotelIdx = spots.findIndex(function(s) {{ return s.title && s.title.includes('住宿'); }});
            var insertAt = bestIdx + 1;
            if (hotelIdx >= 0 && insertAt > hotelIdx) insertAt = hotelIdx;
            return insertAt;
        }}

        // 即時預覽插入位置
        function previewInsert() {{
            var url = document.getElementById('custom-url').value;
            var day = parseInt(document.getElementById('custom-day').value);
            var preview = document.getElementById('custom-preview');
            var coord = parseGoogleMapsUrl(url);
            if (!coord || !tourData[day]) {{ preview.textContent = ''; return; }}
            // 台灣範圍檢查
            if (coord.lat < 21 || coord.lat > 26 || coord.lon < 118 || coord.lon > 123) {{
                preview.style.color = '#e53935';
                preview.textContent = '⚠ 座標不在台灣範圍內';
                return;
            }}
            var idx = findInsertIndex(day, coord.lat, coord.lon);
            var spots = tourData[day].spots;
            var afterName = idx > 0 ? spots[idx-1].name : '起點';
            preview.style.color = '#1976D2';
            preview.textContent = '→ 插在「' + afterName + '」後面（第 ' + (idx+1) + ' 站）';
        }}

        // 新增景點主函式
        function addCustomSpot() {{
            var url  = document.getElementById('custom-url').value.trim();
            var name = document.getElementById('custom-name').value.trim() || '自訂景點';
            var day  = parseInt(document.getElementById('custom-day').value);
            var msg  = document.getElementById('custom-msg');

            // 短網址提示
            if (url.includes('goo.gl') || url.includes('maps.app')) {{
                msg.style.color = '#e65100';
                msg.textContent = '⚠ 短網址請先在瀏覽器開啟，再複製網址列的完整連結';
                return;
            }}

            var coord = parseGoogleMapsUrl(url);
            if (!coord) {{
                msg.style.color = '#e53935';
                msg.textContent = '✗ 無法解析座標，請確認連結格式';
                return;
            }}
            if (coord.lat < 21 || coord.lat > 26 || coord.lon < 118 || coord.lon > 123) {{
                msg.style.color = '#e53935';
                msg.textContent = '✗ 座標不在台灣範圍，請確認連結正確';
                return;
            }}

            var spots = tourData[day].spots;
            var insertAt = findInsertIndex(day, coord.lat, coord.lon);

            // 產生唯一 spot_id（供 AI 快取與 marker 登錄使用）
            var customSpotId = 'custom_' + Date.now();

            // 插入新景點
            spots.splice(insertAt, 0, {{
                lat:      coord.lat,
                lon:      coord.lon,
                name:     name,
                title:    '第' + (insertAt + 1) + '站：自訂',
                nav:      'https://www.google.com/maps/dir/?api=1&destination=' + coord.lat + ',' + coord.lon,
                icon:     'star',
                color:    'pink',
                desc:     '使用者自訂景點',
                img:      '',
                spot_id:  customSpotId,
                spot_type:'景點'
            }});

            // 重新編號所有站
            spots.forEach(function(s, i) {{
                if (s.title) s.title = s.title.replace(new RegExp('^第[0-9]+站'), '第' + (i+1) + '站');
            }});

            // 重新渲染當天地圖與卡片
            switchDay(day);

            // 觸發 AI 導覽（背景非同步，完成後自動更新 popup）
            fetchAiDesc({{ spot_id: customSpotId, name: name, spot_type: '景點' }});

            // 清空輸入
            document.getElementById('custom-url').value = '';
            document.getElementById('custom-name').value = '';
            document.getElementById('custom-preview').textContent = '';
            msg.style.color = '#2e7d32';
            msg.textContent = '✓ 已加入第 ' + day + ' 天第 ' + (insertAt+1) + ' 站！AI 導覽產生中…';

            // 3秒後清除提示
            setTimeout(function() {{ msg.textContent = ''; }}, 3000);
        }}

        // ==========================================
        // Gemini AI 描述快取 & Marker 登錄表
        // aiDescCache: spot_id -> AI 文字（已取得）
        // markerRegistry: spot_id -> Leaflet marker 物件
        // ==========================================
        var aiDescCache = {{}};    // spot_id -> desc string
        var markerRegistry = {{}}; // spot_id -> L.marker

        // ==========================================
        // 原有行程切換邏輯
        // ==========================================
        function buildPopupContent(spot, aiDesc) {{
            var imgHtml = spot.img
                ? `<img src="${{spot.img}}" style="width:100%;height:120px;object-fit:cover;border-radius:4px;margin-bottom:5px;">`
                : '';
            var aiHtml;
            if (aiDesc) {{
                aiHtml = `<div style="font-size:11px;color:#1565C0;margin-top:5px;line-height:1.5;">✨ ${{aiDesc}}</div>`;
            }} else {{
                aiHtml = `<div style="font-size:11px;color:#1565C0;margin-top:5px;line-height:1.5;"><span class="ai-loading">✨ AI 導覽載入中…</span></div>`;
            }}
            return `<div style="width:220px;">
                        ${{imgHtml}}
                        <b>${{spot.title}}</b><br>
                        <span style="font-size:14px;font-weight:bold;">${{spot.name}}</span><br>
                        <span style="font-size:11px;color:#666;">${{spot.desc}}</span>
                        ${{aiHtml}}
                    </div>`;
        }}

        function switchDay(dayNum) {{
            if (!tourData[dayNum]) return;
            var data = tourData[dayNum];

            document.querySelectorAll('.day-btn').forEach(b => b.classList.remove('active'));
            var btn = document.getElementById('btn_day_' + dayNum);
            if(btn) btn.classList.add('active');

            currentLayerGroup.clearLayers();
            markerRegistry = {{}};  // 清除舊天的 marker 登錄

            if (data.route && data.route.length > 0) {{
                L.polyline(data.route, {{color: data.color, weight: 5, opacity: 0.8}}).addTo(currentLayerGroup);
            }}

            data.spots.forEach(spot => {{
                var myIcon = L.AwesomeMarkers.icon({{
                    icon: spot.icon, markerColor: spot.color, prefix: 'fa', iconColor: 'white'
                }});

                // 建立 popup 時，若快取已有 AI 文字直接填入，否則顯示「載入中」
                var marker = L.marker([spot.lat, spot.lon], {{icon: myIcon}})
                    .bindPopup(buildPopupContent(spot, aiDescCache[spot.spot_id] || null))
                    .addTo(currentLayerGroup);

                // 【修正】每次 popup 被打開時，用最新快取重新渲染內容
                // 這樣無論 AI 是在 popup 開啟前還是後回來，都能正確顯示
                marker.on('popupopen', (function(s) {{
                    return function() {{
                        this.setPopupContent(buildPopupContent(s, aiDescCache[s.spot_id] || null));
                    }};
                }})(spot));

                markerRegistry[spot.spot_id] = marker;
            }});

            var container = document.getElementById('panel_container');
            container.innerHTML = '';

            if (data.spots.length > 0) {{
                mapObject.flyTo([data.spots[0].lat, data.spots[0].lon], 13);
            }}

            data.spots.forEach(spot => {{
                var imgStyle = spot.img ? `background-image: url('${{spot.img}}');` : 'background: #eee;';

                var card = `
                <div class="card" style="border-left-color: ${{spot.color}}">
                    <div class="card-img" style="${{imgStyle}}"></div>
                    <div class="card-info" onclick="mapObject.flyTo([${{spot.lat}}, ${{spot.lon}}], 16)" title="點擊地圖圖示可看 AI 導覽">
                        <div style="font-size:11px; font-weight:bold; color:${{spot.color}}">${{spot.title}}</div>
                        <div style="font-size:13px; font-weight:bold; color:#333;">${{spot.name}}</div>
                        <div style="font-size:10px; color:#aaa; margin-top:2px;">📍 點地圖查看 AI 導覽</div>
                    </div>
                    <a href="${{spot.nav}}" target="_blank" class="nav-btn">GO</a>
                </div>
                `;
                container.innerHTML += card;
            }});
        }}

        // ==========================================
        // Gemini AI 背景逐一載入
        // ==========================================

        // 【修正】applyAiDesc 改為透過 Leaflet marker API 更新 popup 內容，
        // 不再用 getElementById（popup 未開啟時 DOM 不存在，寫入會靜默失敗）
        function applyAiDesc(spotId, desc) {{
            aiDescCache[spotId] = desc;
            // 若此 marker 目前在地圖上（同一天），更新它的 popup content
            var marker = markerRegistry[spotId];
            if (marker) {{
                // 找到對應的 spot 物件，重新組裝 popup
                for (var day in tourData) {{
                    var found = tourData[day].spots.find(function(s) {{ return s.spot_id === spotId; }});
                    if (found) {{
                        var newContent = buildPopupContent(found, desc);
                        marker.setPopupContent(newContent);
                        break;
                    }}
                }}
            }}
        }}

        async function fetchAiDesc(spot) {{
            if (aiDescCache[spot.spot_id]) {{
                // 快取命中：確保 marker popup 是最新狀態（例如換天後重建的 marker）
                applyAiDesc(spot.spot_id, aiDescCache[spot.spot_id]);
                return;
            }}
            try {{
                var resp = await fetch('/ai_desc', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        name:      spot.name,
                        city:      aiCity,
                        spot_type: spot.spot_type || '景點'
                    }})
                }});
                var data = await resp.json();
                var desc = data.desc || '';
                applyAiDesc(spot.spot_id, desc);
            }} catch(e) {{
                applyAiDesc(spot.spot_id, '（導覽暫時無法載入）');
            }}
        }}

        async function loadAllAiDescs() {{
            // 逐一呼叫，每次間隔 800ms 避免超過免費額度速率限制
            for (var day in tourData) {{
                var spots = tourData[day].spots;
                for (var i = 0; i < spots.length; i++) {{
                    await fetchAiDesc(spots[i]);
                    await new Promise(r => setTimeout(r, 800));
                }}
            }}
        }}

        // 頁面載入後由 window.onload 呼叫 switchDay(1) 與 loadAllAiDescs()
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(html_ui))
    return m.get_root().render()

# ==========================================
# 6. Flask Routes
# ==========================================

@app.route('/')
def index():
    cities = []
    if not df_attr.empty:
        cities = sorted(list(set(
            df_attr['PostalAddress']
            .apply(lambda x: x.get('City') if isinstance(x, dict) else None)
            .dropna()
        )))
    styles = list(STYLE_MAP.keys())
    return render_template('index.html', cities=cities, styles=styles)


@app.route('/generate', methods=['POST'])
def generate():
    city      = request.form.get('city', '宜蘭縣')
    style     = request.form.get('style', ' 不限風格 (隨機)')
    days      = int(request.form.get('days', 2))
    spots     = int(request.form.get('spots', 3))

    map_html = generate_multi_day_map(city, days, spots, style)

    # 若資料不足回傳錯誤訊息（字串不含 <html>）
    if not map_html or not map_html.strip().startswith('<!'):
        return render_template('index.html',
                               cities=sorted(list(set(
                                   df_attr['PostalAddress']
                                   .apply(lambda x: x.get('City') if isinstance(x, dict) else None)
                                   .dropna()
                               ))),
                               styles=list(STYLE_MAP.keys()),
                               error=map_html or '資料不足，請換個城市試試')

    # 暫存地圖存成檔案（worker 間共用，重啟前不消失）
    map_id = str(uuid.uuid4())
    map_path = os.path.join(MAP_CACHE_DIR, f'{map_id}.html')
    with open(map_path, 'w', encoding='utf-8') as f:
        f.write(map_html)

    return render_template('result.html',
                           map_id=map_id,
                           city=city,
                           days=days,
                           spots=spots,
                           style=style)


@app.route('/map/<map_id>')
def serve_map(map_id):
    # 安全性：只允許 uuid 格式，防止路徑穿越
    import re
    if not re.match(r'^[0-9a-f-]{36}$', map_id):
        return '無效的地圖 ID', 400
    map_path = os.path.join(MAP_CACHE_DIR, f'{map_id}.html')
    if not os.path.exists(map_path):
        return '地圖已過期，請重新規劃', 404
    with open(map_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/ai_desc', methods=['POST'])
def ai_desc():
    """接收景點名稱與城市，回傳 Gemini 生成的導覽介紹"""
    data     = request.get_json()
    name     = data.get('name', '')
    city     = data.get('city', '')
    spot_type = data.get('spot_type', '景點')  # 景點 / 餐廳 / 住宿

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return jsonify({'desc': '（AI 導覽未啟用）'})

    prompt = (
        f"你是台灣在地旅遊導覽員，用繁體中文寫一段關於「{name}」的簡短介紹。"
        f"這是位於{city}的{spot_type}。"
        f"字數控制在 50 字左右，語氣活潑，突顯特色，不要用條列式。"
    )

    try:
        url = (
            'https://generativelanguage.googleapis.com/v1beta/models/'
            f'gemini-2.5-flash:generateContent?key={api_key}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'maxOutputTokens': 1500, 'temperature': 0.7}
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        desc = (result['candidates'][0]['content']['parts'][0]['text']).strip()
        return jsonify({'desc': desc})
    except Exception as e:
        return jsonify({'desc': '（導覽載入失敗）', 'error': str(e)}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

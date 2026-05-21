import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# ESPN API endpoint for injury data
ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Fallback HTML URLs in case API fails
FALLBACK_URLS = [
    "https://www.espn.com/nba/injuries",
    "https://www.espn.com/nba/injuries/_/type/injury-report", 
    "https://www.espn.com/nba/players/injuries",
]

OUTPUT_PATH = "sports_betting/data/injuries/injuries.json"

# Try different user agents in case ESPN is blocking certain ones
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
]

def get_headers(user_agent):
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def get_api_headers():
    """Headers for ESPN API requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }

def fetch_espn_api_injuries():
    """Fetch injury data from ESPN API."""
    print(f"[ESPN API] Fetching data from: {ESPN_API_URL}")
    
    try:
        headers = get_api_headers()
        response = requests.get(ESPN_API_URL, headers=headers, timeout=30)
        print(f"[ESPN API] Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[ESPN API] Failed with status {response.status_code}")
            return {}
            
        data = response.json()
        print(f"[ESPN API] Successfully parsed JSON response")
        
        # Debug: Print the top-level keys
        if isinstance(data, dict):
            print(f"[ESPN API DEBUG] Top-level keys: {list(data.keys())}")
        else:
            print(f"[ESPN API DEBUG] Response is not a dict: {type(data)}")
            return {}
        
        # Check for the 'injuries' key as mentioned in the issue
        if 'injuries' in data:
            injuries_data = data['injuries']
            print(f"[ESPN API DEBUG] Found 'injuries' key with type: {type(injuries_data)}")
            
            if isinstance(injuries_data, list):
                print(f"[ESPN API DEBUG] 'injuries' is a list with {len(injuries_data)} items")
                if injuries_data:
                    # Print structure of first 2 items for debugging as requested
                    print("[ESPN API DEBUG] === Structure of first 2 items ===")
                    for i, item in enumerate(injuries_data[:2]):
                        print(f"[ESPN API DEBUG] Item {i+1}:")
                        if isinstance(item, dict):
                            print(f"[ESPN API DEBUG]   Keys: {list(item.keys())}")
                            # Print full structure for debugging
                            import json
                            print(f"[ESPN API DEBUG]   Full structure: {json.dumps(item, indent=4)}")
                        else:
                            print(f"[ESPN API DEBUG]   Type: {type(item)}, Value: {item}")
                        print("[ESPN API DEBUG] " + "="*50)
            elif isinstance(injuries_data, dict):
                print(f"[ESPN API DEBUG] 'injuries' is a dict with keys: {list(injuries_data.keys())}")
                # Print full structure for debugging
                import json
                print(f"[ESPN API DEBUG] Full dict structure: {json.dumps(injuries_data, indent=4)}")
            else:
                print(f"[ESPN API DEBUG] 'injuries' has unexpected type: {type(injuries_data)}")
            
            # Parse the injuries data
            return parse_api_injuries_data(injuries_data)
        else:
            print("[ESPN API DEBUG] 'injuries' key not found in response")
            # Print available keys for debugging
            available_keys = list(data.keys()) if isinstance(data, dict) else []
            print(f"[ESPN API DEBUG] Available keys: {available_keys}")
            return {}
            
    except requests.RequestException as e:
        print(f"[ESPN API] Request failed: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ESPN API] Failed to parse JSON: {e}")
        return {}
    except Exception as e:
        print(f"[ESPN API] Unexpected error: {e}")
        return {}

def parse_api_injuries_data(injuries_data):
    """Parse injury data from ESPN API response."""
    injuries = {}
    for item in injuries_data:
        if not isinstance(item, dict):
            continue
        team_name = item.get('displayName')
        if not team_name:
            continue
        team_key = team_name.lower()
        player_list = item.get('injuries', [])
        for player in player_list:
            athlete = player.get('athlete', {})
            player_name = athlete.get('displayName')
            status = player.get('status', 'out')
            if player_name:
                if team_key not in injuries:
                    injuries[team_key] = {}
                injuries[team_key][player_name] = str(status).lower()
    print(f"[ESPN API] Parsed {len(injuries)} teams with injuries")
    return injuries

def _safe_team_name(table) -> str:
    """Extract the team name safely."""
    section = table.find_parent("section")
    if section:
        heading = section.find(["h2", "h3"])
        if heading:
            return heading.get_text(" ", strip=True).lower()
        aria_label = section.get("aria-label")
        if aria_label:
            return str(aria_label).strip().lower()
    label = table.find_previous(["h2", "h3", "span"])
    if label:
        return label.get_text(" ", strip=True).lower()
    caption = table.find("caption")
    if caption:
        return caption.get_text(" ", strip=True).lower()
    return "unknown"

def extract_from_json_data(content):
    """Try to extract injury data from JSON embedded in the page."""
    print("[ESPN DEBUG] Attempting to extract from embedded JSON...")
    injuries = {}
    
    # Look for various ESPN data patterns
    json_patterns = [
        r'window\.__espnfitt__\s*=\s*(\{.*?\});',
        r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});', 
        r'"injuries"\s*:\s*(\[.*?\])',
        r'"injuryData"\s*:\s*(\{.*?\})',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        print(f"[ESPN DEBUG] JSON pattern '{pattern[:30]}...' found {len(matches)} matches")
        
        for match in matches:
            try:
                # Clean up the match and try to parse as JSON
                json_str = match.strip().rstrip(';')
                data = json.loads(json_str)
                print(f"[ESPN DEBUG] Successfully parsed JSON with keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                
                # Try to find injury data in the JSON structure
                injury_data = _extract_injuries_from_json(data)
                if injury_data:
                    injuries.update(injury_data)
                    print(f"[ESPN DEBUG] Extracted {len(injury_data)} teams from JSON")
                    
            except json.JSONDecodeError as e:
                print(f"[ESPN DEBUG] Failed to parse JSON: {e}")
                continue
    
    return injuries

def _extract_injuries_from_json(data, path=""):
    """Recursively search for injury data in JSON structure."""
    injuries = {}
    
    if isinstance(data, dict):
        # Look for injury-related keys
        injury_keys = ['injuries', 'injuryData', 'players', 'teams', 'roster']
        for key in injury_keys:
            if key in data:
                print(f"[ESPN DEBUG] Found injury key '{key}' at path '{path}'")
                sub_data = data[key]
                if isinstance(sub_data, list):
                    for item in sub_data:
                        if isinstance(item, dict):
                            team_name = item.get('team', {}).get('displayName') or item.get('teamName') or 'unknown'
                            player_injuries = item.get('injuries', []) or item.get('players', [])
                            if player_injuries:
                                injuries[team_name.lower()] = {}
                                for injury in player_injuries:
                                    if isinstance(injury, dict):
                                        player = injury.get('athlete', {}).get('displayName') or injury.get('name', 'unknown')
                                        status = injury.get('status', 'out')
                                        injuries[team_name.lower()][player] = status
        
        # Recursively search other keys
        for key, value in data.items():
            if key not in ['__typename', '_id'] and isinstance(value, (dict, list)):
                sub_injuries = _extract_injuries_from_json(value, f"{path}.{key}" if path else key)
                injuries.update(sub_injuries)
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            sub_injuries = _extract_injuries_from_json(item, f"{path}[{i}]")
            injuries.update(sub_injuries)
    
    return injuries

def fetch_espn_injuries():
    """Fetch injury data from ESPN with API first, then HTML fallback approaches."""
    injuries = {}
    
    # First try the ESPN API
    print("[ESPN] Trying ESPN API first...")
    api_injuries = fetch_espn_api_injuries()
    if api_injuries:
        print(f"[ESPN] Successfully got data from API: {len(api_injuries)} teams")
        injuries.update(api_injuries)
        return injuries
    else:
        print("[ESPN] API failed, falling back to HTML scraping...")
    
    # Fallback to HTML scraping with multiple URLs and user agents
    for url_idx, url in enumerate(FALLBACK_URLS):
        print(f"[ESPN HTML] Trying URL {url_idx + 1}: {url}")
        
        for ua_idx, user_agent in enumerate(USER_AGENTS):
            headers = get_headers(user_agent)
            print(f"[ESPN HTML] Attempt {ua_idx + 1} with User-Agent: {user_agent[:50]}...")
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                print(f"[ESPN HTML] Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"[ESPN HTML] Non-200 status, trying next...")
                    continue
                    
                content = response.text
                print(f"[ESPN HTML] Page content length: {len(content)} characters")
                
                # Check if we got blocked (common blocking responses)
                if any(block_indicator in content.lower() for block_indicator in 
                       ['access denied', 'blocked', 'cloudflare', 'security check', 'rate limited']):
                    print(f"[ESPN HTML] Appears to be blocked, trying next approach...")
                    continue
                
                # First try to parse as traditional HTML tables
                table_injuries = _parse_html_tables(content)
                if table_injuries:
                    print(f"[ESPN HTML] Successfully extracted {len(table_injuries)} teams from HTML tables")
                    injuries.update(table_injuries)
                    break
                
                # If no tables found, try JSON extraction
                json_injuries = extract_from_json_data(content)
                if json_injuries:
                    print(f"[ESPN HTML] Successfully extracted {len(json_injuries)} teams from JSON data")
                    injuries.update(json_injuries)
                    break
                
                # If still no data, try modern CSS selectors
                modern_injuries = _parse_modern_structure(content)
                if modern_injuries:
                    print(f"[ESPN HTML] Successfully extracted {len(modern_injuries)} teams from modern structure")
                    injuries.update(modern_injuries)
                    break
                    
                print(f"[ESPN HTML] No injury data found with this URL/user-agent combination")
                
            except requests.RequestException as e:
                print(f"[ESPN HTML] Request failed: {e}")
                continue
        
        if injuries:
            break  # Found data, stop trying other URLs
    
    if not injuries:
        print(f"[ESPN WARNING] No injury data found after trying API and all HTML fallback URLs")
    
    # Remove empty teams
    print(f"[ESPN DEBUG] Before filtering: {len(injuries)} teams")
    for team, players in injuries.items():
        print(f"[ESPN DEBUG]   {team}: {len(players)} players")
    
    injuries = {team: players for team, players in injuries.items() if players}
    print(f"[ESPN] Found {len(injuries)} teams with injuries")
    
    # Final debug summary
    total_players = sum(len(players) for players in injuries.values())
    print(f"[ESPN DEBUG] Final result: {len(injuries)} teams, {total_players} total injured players")
    
    return injuries

def _parse_html_tables(content):
    """Parse traditional HTML table structure."""
    print(f"[ESPN DEBUG] Parsing HTML tables...")
    soup = BeautifulSoup(content, "html.parser")
    injuries = {}
    tables = soup.find_all("table")
    print(f"[ESPN DEBUG] Found {len(tables)} table elements")

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        print(f"[ESPN DEBUG] Table {i+1}: {len(rows)} rows total")
        
        if not rows:
            continue
            
        data_rows = rows[1:]  # Skip header row
        team = _safe_team_name(table)
        print(f"[ESPN DEBUG] Table {i+1} team: '{team}'")
        
        if team not in injuries:
            injuries[team] = {}

        player_count = 0
        for j, row in enumerate(data_rows):
            cols = row.find_all("td")
            print(f"[ESPN DEBUG]   Row {j+1}: {len(cols)} columns")
            
            if len(cols) < 2:  # Need at least player and status
                print(f"[ESPN DEBUG]   Row {j+1}: Skipped (insufficient columns)")
                continue

            player = cols[0].get_text(" ", strip=True)
            # Status might be in different columns depending on table structure
            status = "out"  # Default
            if len(cols) >= 3:
                status = cols[2].get_text(" ", strip=True)
            elif len(cols) >= 2:
                status = cols[1].get_text(" ", strip=True)
                
            print(f"[ESPN DEBUG]   Row {j+1}: Player='{player}', Status='{status}'")

            if player:
                injuries[team][player] = status
                player_count += 1
        
        print(f"[ESPN DEBUG] Table {i+1}: Added {player_count} players for team '{team}'")
    
    return injuries

def _parse_modern_structure(content):
    """Try to parse modern div-based layouts."""
    print(f"[ESPN DEBUG] Parsing modern structure...")
    soup = BeautifulSoup(content, "html.parser")
    injuries = {}
    
    # Look for injury-related divs or sections
    injury_selectors = [
        'div[class*="injury"]',
        'div[class*="player"]', 
        'section[class*="injury"]',
        '.injury-report',
        '.player-injury',
        '[data-testid*="injury"]',
    ]
    
    for selector in injury_selectors:
        elements = soup.select(selector)
        print(f"[ESPN DEBUG] Selector '{selector}' found {len(elements)} elements")
        
        if elements:
            # Try to extract injury data from these modern elements
            for element in elements:
                team_name = _extract_team_from_modern_element(element)
                player_data = _extract_players_from_modern_element(element)
                
                if team_name and player_data:
                    injuries[team_name.lower()] = player_data
    
    return injuries

def _extract_team_from_modern_element(element):
    """Extract team name from modern HTML element."""
    # Try various ways to find team name
    team_selectors = ['h1', 'h2', 'h3', '.team-name', '[class*="team"]']
    
    for selector in team_selectors:
        team_elem = element.find(selector) or element.find_parent().find(selector) if element.find_parent() else None
        if team_elem:
            return team_elem.get_text(strip=True)
    
    return "unknown"

def _extract_players_from_modern_element(element):
    """Extract player injury data from modern HTML element."""
    players = {}
    
    # Look for player name patterns
    player_elements = element.find_all(['div', 'span'], text=True)
    
    for elem in player_elements:
        text = elem.get_text(strip=True)
        # Simple heuristic: if text looks like a name (2+ words, capitalized)
        if len(text.split()) >= 2 and text[0].isupper():
            # Look for status nearby
            status = "out"
            next_elem = elem.find_next_sibling()
            if next_elem:
                status_text = next_elem.get_text(strip=True).lower()
                if any(word in status_text for word in ['out', 'questionable', 'doubtful', 'probable']):
                    status = status_text
            
            players[text] = status
    
    return players

def run_injury_pipeline():
    """Run the injury scraping and save to JSON."""
    injuries = fetch_espn_injuries()
    
    # If main scraping failed, try alternative sources
    if not injuries:
        print("[ESPN] Main scraping returned no data, trying alternative sources...")
        try:
            from .alternative_sources import get_fallback_injury_data
            injuries = get_fallback_injury_data()
            if injuries:
                print(f"[ESPN] Alternative sources found data for {len(injuries)} teams")
            else:
                print("[ESPN] Alternative sources also failed")
        except ImportError:
            print("[ESPN] Alternative sources not available")
        except Exception as e:
            print(f"[ESPN] Alternative sources failed: {e}")
    
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(injuries, f, indent=2)
    print(f"[ESPN] Saved injuries to {output_path}")

def test_parser():
    """Test the parser with sample data structure."""
    print("\n=== Testing parser with sample data ===")
    
    # Sample data structure based on typical ESPN API
    sample_injuries = [
        {
            "team": {
                "displayName": "Los Angeles Lakers",
                "name": "Lakers",
                "abbreviation": "LAL"
            },
            "athlete": {
                "displayName": "LeBron James",
                "name": "LeBron James"
            },
            "status": {
                "type": "Out",
                "displayName": "Out"
            }
        },
        {
            "team": {
                "displayName": "Golden State Warriors",
                "name": "Warriors"
            },
            "athlete": {
                "displayName": "Stephen Curry",
                "name": "Stephen Curry"
            },
            "status": "Questionable"
        }
    ]
    
    result = parse_api_injuries_data(sample_injuries)
    print(f"\nTest result: {result}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_parser()
    else:
        run_injury_pipeline()

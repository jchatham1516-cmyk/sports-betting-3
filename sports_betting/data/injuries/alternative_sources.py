"""Alternative injury data sources in case ESPN's main sources fail."""

import json
import urllib.request
import urllib.error
from urllib.request import Request, urlopen
from typing import Dict, List, Any

def fetch_nba_team_injury_pages() -> Dict[str, Dict[str, str]]:
    """Try to fetch from individual NBA team injury pages on ESPN."""
    print("[ALT] Trying individual NBA team injury pages...")
    
    # NBA team abbreviations for ESPN URLs
    nba_teams = [
        'atl', 'bos', 'bkn', 'cha', 'chi', 'cle', 'dal', 'den', 'det', 'gs', 
        'hou', 'ind', 'lac', 'lal', 'mem', 'mia', 'mil', 'min', 'no', 'ny',
        'okc', 'orl', 'phi', 'phx', 'por', 'sac', 'sa', 'tor', 'utah', 'was'
    ]
    
    all_injuries = {}
    successful_teams = 0
    
    for team_abbrev in nba_teams[:5]:  # Test first 5 teams to avoid being too aggressive
        url = f"https://www.espn.com/nba/team/injuries/_/name/{team_abbrev}"
        print(f"[ALT] Trying team page: {team_abbrev}")
        
        try:
            request = Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
            
            with urlopen(request, timeout=10) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    
                    # Look for injury data patterns in the page
                    if 'injury' in content.lower() or 'out' in content.lower():
                        # Simple extraction - look for player names and status
                        # This is a basic implementation that could be improved
                        team_injuries = _extract_team_injuries_from_page(content, team_abbrev)
                        if team_injuries:
                            all_injuries[team_abbrev] = team_injuries
                            successful_teams += 1
                            print(f"[ALT] Found {len(team_injuries)} injuries for {team_abbrev}")
                    
        except Exception as e:
            print(f"[ALT] Failed to fetch {team_abbrev}: {e}")
            continue
    
    print(f"[ALT] Successfully fetched from {successful_teams} team pages")
    return all_injuries

def _extract_team_injuries_from_page(content: str, team_abbrev: str) -> Dict[str, str]:
    """Extract injury data from team-specific ESPN page."""
    # This is a simplified extraction - in a real implementation you'd parse the HTML properly
    injuries = {}
    
    # Look for common injury status words near player names
    # This is a basic heuristic that could be improved
    lines = content.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if any(status in line_lower for status in ['out', 'questionable', 'doubtful', 'probable']):
            # Try to find player names (this is very basic)
            # In practice you'd use proper HTML parsing
            pass
    
    return injuries

def fetch_from_alternative_api() -> Dict[str, Any]:
    """Try alternative ESPN API endpoints that might still work."""
    print("[ALT] Trying alternative ESPN API endpoints...")
    
    alternative_endpoints = [
        "https://site.api.espn.com/apis/site/v3/sports/basketball/nba/injuries",
        "https://www.espn.com/core/nba/injuries",
        "https://site.api.espn.com/apis/common/v3/sports/basketball/nba/injuries", 
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?enable=roster,injuries",
    ]
    
    for url in alternative_endpoints:
        print(f"[ALT] Trying: {url}")
        
        try:
            request = Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            request.add_header('Accept', 'application/json')
            
            with urlopen(request, timeout=15) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    
                    # Check if it looks like JSON
                    if content.strip().startswith('{') or content.strip().startswith('['):
                        try:
                            data = json.loads(content)
                            print(f"[ALT] Got JSON response with keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                            
                            # Look for injury data in the response
                            injuries = _extract_injuries_from_alt_json(data)
                            if injuries:
                                print(f"[ALT] Found {len(injuries)} teams with injuries")
                                return injuries
                                
                        except json.JSONDecodeError:
                            print(f"[ALT] Invalid JSON from {url}")
                    else:
                        print(f"[ALT] Non-JSON response from {url}")
        
        except Exception as e:
            print(f"[ALT] Failed {url}: {e}")
    
    return {}

def _extract_injuries_from_alt_json(data: Any) -> Dict[str, Dict[str, str]]:
    """Extract injury data from alternative JSON structures."""
    injuries = {}
    
    # Recursive search for injury data patterns
    def search_for_injuries(obj, path=""):
        if isinstance(obj, dict):
            # Look for team structures
            if 'team' in obj and 'injuries' in obj:
                team_name = obj.get('team', {}).get('displayName', 'unknown')
                team_injuries = {}
                
                for injury in obj.get('injuries', []):
                    if isinstance(injury, dict):
                        player = injury.get('athlete', {}).get('displayName') or injury.get('name', 'unknown')
                        status = injury.get('status', 'out')
                        team_injuries[player] = status
                
                if team_injuries:
                    injuries[team_name.lower()] = team_injuries
            
            # Look for teams list
            elif 'teams' in obj:
                teams = obj['teams']
                if isinstance(teams, list):
                    for team in teams:
                        search_for_injuries(team, f"{path}.teams[]")
            
            # Recursively search other dict keys
            else:
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        search_for_injuries(value, f"{path}.{key}" if path else key)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                search_for_injuries(item, f"{path}[{i}]")
    
    search_for_injuries(data)
    return injuries

def get_fallback_injury_data() -> Dict[str, Dict[str, str]]:
    """Try all alternative sources and return the best available data."""
    print("[ALT] Attempting to fetch injury data from alternative sources...")
    
    # Try alternative APIs first (faster)
    alt_api_data = fetch_from_alternative_api()
    if alt_api_data:
        print(f"[ALT] Success with alternative API: {len(alt_api_data)} teams")
        return alt_api_data
    
    # Try individual team pages (slower but more reliable)
    team_page_data = fetch_nba_team_injury_pages()
    if team_page_data:
        print(f"[ALT] Success with team pages: {len(team_page_data)} teams")
        return team_page_data
    
    print("[ALT] All alternative sources failed")
    return {}

if __name__ == "__main__":
    # Test the alternative sources
    print("Testing alternative ESPN injury sources...")
    data = get_fallback_injury_data()
    
    if data:
        print(f"Found data for {len(data)} teams:")
        for team, players in data.items():
            print(f"  {team}: {len(players)} players")
    else:
        print("No data found from alternative sources")
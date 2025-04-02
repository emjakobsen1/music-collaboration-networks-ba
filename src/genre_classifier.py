import re
from collections import defaultdict
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import numpy as np


classification_stats = defaultdict(lambda: {"count": 0, "subgenres": defaultdict(int)})


super_genres = {
    "asian": [
    "desi", "mandopop", "bollywood", "opm", "anime", "k-rap", "p-pop",
    "bhajan", "k-ballad", "cantopop", "bhangra", "chinese hip hop",
    "tamil pop", "vocaloid", "chinese rock", "kollywood", "tamil dance",
    "harana", "pinoy alternative", "tollywood", "japanese classical",
    "gufeng", "chinese r&b", "j-rock", "j-dance", "kundiman",
    "japanese vgm", "chinese indie", "shibuya-kei", "pinoy indie",
    "japanese indie", "indian indie", "mollywood", "hindi indie",
    "taiwanese indie", "kayokyoku", "vietnamese bolero", "tamil indie",
    "indonesian indie", "rkt", "enka", "budots", "dangdut", "luk thung",
    "vietnamese lo-fi", "vietnam indie", "phleng phuea chiwit",
    "bhojiwood", "funkot", "thai indie pop", "funk de bh",
    "southern thai music", "batak"
],
    "folk-world": [
    "indie folk", "folk", "singer-songwriter", "christian folk", "celtic",
    "soca", "new age", "afrobeat", "afrobeats", "azonto", "afropop",
    "sea shanties", "kizomba", "cha cha cha", "acoustic folk",
    "traditional music", "italian singer-songwriter", "latin folk",
    "highlife", "polka", "neofolk", "swedish ballads", "afro soul",
    "rhumba", "schlager", "malay", "afropiano", "raï", "afro adura",
    "gnawa", "arabesk", "fado", "punta", "iskelmä", "schlagerparty",
    "dansktop", "mahraganat", "dansband", "maluku", "singeli",
    "karadeniz folk", "fújì", "manele", "moroccan chaabi",
    "egyptian shaabi", "algerian chaabi"
],
    "electronic": [
    "edm", "dubstep", "trance", "progressive house", "big room", "trip hop",
    "dancehall", "dub", "ragga", "riddim", "future bass", "chillstep",
    "nu disco", "new wave", "nu jazz", "electro house", "drum and bass",
    "ambient", "idm", "trap", "chillwave", "tropical house", "drumstep",
    "house", "future house", "downtempo", "french house", "cedm", "electronica",
    "bass music", "glitch", "deep house", "edm trap", "moombahton", "deathstep",
    "lounge", "bass house", "tech house", "liquid funk", "lo-fi beats",
    "melodic bass", "electronic", "alternative dance", "minimal techno",
    "electro swing", "lo-fi", "jungle", "synthwave", "melodic house",
    "electroclash", "funky house", "melbourne bounce", "hardstyle", "electro",
    "disco house", "new rave", "melodic techno", "indie dance", "techno",
    "g-house", "ebm", "eurodance", "vaporwave", "cold wave", "dance",
    "chicago house", "breakbeat", "big beat", "witch house", "italo dance",
    "dark trap", "miami bass", "bassline", "jazz beats", "psytrance", "afroswing",
    "tribal house", "space music", "piano house", "latin house", "frenchcore",
    "uk funky", "exotica", "slap house", "acid house", "bounce", "funk melody",
    "organic house", "afro house", "lo-fi house", "progressive trance", "kuduro", "indie electronic",
    "dub techno", "dark ambient", "chill house", "latin dance", "jersey club",
    "hip house", "entehno", "nightcore", "gabber", "acid techno", "hardcore techno",
    "rave", "brazilian bass", "stutter house", "amapiano", "hard house",
    "italian trap", "russelåter", "hard techno", "gqom", "breakcore",
    "brazilian trap", "visualbrazilian trap", "tekno", "vinahouse", "epadunk",
    "trap funk", "thai trap", "chilean trap", "argentine trap", "tecnobrega", "jazz house", 
    "hypertechno", "baltimore club", "rally house"
],
    "caribbean-latin-america": [
    "reggae", "norteño", "banda", "música mexicana", "corrido", "grupera",
    "roots reggae", "reggaeton", "latin", "salsa", "ska", "merengue",
    "cumbia norteña", "ranchera", "cumbia", "mariachi", "tejano",
    "latin alternative", "sierreño", "urbano latino", "bachata",
    "salsa romantica", "rocksteady", "lovers rock", "trap latino",
    "corridos bélicos", "mpb", "son cubano", "champeta", "zouk",
    "cumbia sonidera", "bolero", "samba", "norteño-sax", "duranguense",
    "mambo", "flamenco", "nz reggae", "música tropical", "calypso",
    "trova", "tango", "dembow", "spanish-language reggae", "timba",
    "nueva trova", "vallenato", "pentecostal", "mexican indie",
    "mexican ska", "sertanejo", "sertanejo universitário", "sad sierreño",
    "brazilian rock", "corridos tumbados", "nova mpb", "arrocha", "pagode",
    "bongo flava", "brazilian funk", "axé", "alté", "shatta", "guaracha",
    "funk carioca", "forró", "cuarteto", "agronejo", "chicha",
    "pagode baiano", "sertanejo tradicional", "latin afrobeat",
    "electro corridos", "piseiro", "brega", "neoperreo",
    "folklore argentino", "huayno", "reggaeton chileno",
    "chilean mambo", "flamenco urbano", "oyun havasi", "forro tradicional",
    "neomelodico", "turreo", "funk consciente", "seresta",
    "reggaeton mexa", "punto guajiro", "latin afrobeats", "brega funk"
],
    "classical": [
    "classical", "orchestra", "classical piano", "opera", "chamber music",
    "neoclassical", "choral", "requiem", "minimalism", "chanson",
    "medieval", "gregorian chant", "canzone napoletana"
],
    "country": [
    "country", "americana", "alt country", "bluegrass", "acoustic country",
    "red dirt", "texas country", "newgrass", "traditional country",
    "christian country", "gothic country", "pop country", "country blues"
],
    "jazz-blues": [
    "jazz", "big band", "vocal jazz", "swing music", "cool jazz", "jazz blues",
    "adult standards", "boogie-woogie", "honky tonk", "hard bop", "blues",
    "smooth jazz", "bebop", "jazz funk", "modern blues", "rockabilly",
    "latin jazz", "jazz fusion", "acid jazz", "bossa nova", "classic blues",
    "free jazz", "french jazz", "ragtime", "funk", "brazilian jazz",
    "indie jazz", "go-go", "boogie", "ethiopian jazz"
],
    "r&b": [
    "r&b", "soul", "quiet storm", "neo soul", "motown", "gospel r&b",
    "northern soul", "new jack swing", "doo-wop", "alternative r&b", "smooth r&b",
    "soul blues", "retro soul", "philly soul", "soul jazz", "smooth soul",
    "pop soul", "indie r&b", "pinoy r&b", "j-r&b", "latin r&b", "pop r&b"
],
    "hip-hop": [
    "rap", "hip hop", "southern hip hop", "east coast hip hop", "west coast hip hop",
    "gangster rap", "g-funk", "underground hip hop", "jazz rap", "crunk",
    "christian hip hop", "hyphy", "country hip hop", "hardcore hip hop",
    "latin hip hop", "alternative hip hop", "experimental hip hop", "boom bap",
    "memphis rap", "melodic rap", "mexican hip hop", "cloud rap", "emo rap",
    "drill", "grime", "hiplife", "nerdcore", "freestyle", "punk rap",
    "brazilian hip hop", "meme rap", "german hip hop", "desi hip hop",
    "punjabi hip hop", "rage rap", "norwegian rap", "tamil hip hop",
    "pinoy hip hop", "hindi hip hop", "j-rap", "uk drill", "asakaa",
    "brooklyn drill", "turkish hip hop", "sexy drill", "thai hip hop",
    "finnish hip hop", "portuguese hip hop", "arabic rap", "gengetone",
    "phonk", "drift phonk", "new york drill", "aussie drill", "anime rap",
    "egyptian hip hop", "vietnamese hip hop", "malay rap", "malayalam hip hop", "norwegian hip hop"
],
    "rock": [
    "metal", "rock", "emo", "pop punk", "classic rock", "christian alternative rock",
    "post-hardcore", "metalcore", "punk", "christian rock", "alternative metal",
    "screamo", "hard rock", "nu metal", "post-grunge", "psychedelic rock",
    "alternative rock", "indie", "reggae rock", "chamber pop", "rap metal",
    "garage rock", "baroque pop", "skate punk", "heavy metal",
    "indie rock", "southern rock", "hardcore punk", "folk rock", "country rock",
    "post-rock", "rock en español", "soft rock", "hardcore", "midwest emo",
    "ska punk", "deathcore", "roots rock", "surf rock", "math rock", "blues rock",
    "glam rock", "post-punk", "art rock", "shoegaze", "progressive rock", "glam metal",
    "death metal", "proto-punk", "grunge", "folk punk", "djent", "latin rock",
    "progressive metal", "melodic hardcore", "industrial", "acid rock", "darkwave",
    "horrorcore", "funk rock", "symphonic rock", "industrial rock", "mexican rock",
    "industrial metal", "space rock", "rap rock", "album rock", "emocore",
    "celtic rock", "rock and roll", "groove metal", "stoner rock", "noise rock",
    "electronic rock", "thrash metal", "melodic metal", "slowcore", "sludge metal",
    "modern rock", "speed metal", "power metal", "dark cabaret", "doom metal",
    "melodic death metal", "madchester", "folk metal", "symphonic metal",
    "psychobilly", "latin indie", "acoustic rock", "argentine rock", "grindcore",
    "deathrock", "gothic metal", "indie punk", "pirate metal", "laïko",
    "happy hardcore", "krautrock", "uk garage", "black metal", "rock urbano",
    "pinoy rock", "medieval metal", "arena rock", "finnish rock", "k-rock",
    "speedcore", "neue deutsche welle", "anatolian rock", "visual kei", "pop rock",
    "indorock", "bisrock", "thai rock", "thai indie rock"
]
,
    "pop": [
    "emo pop", "pop worship", "christian pop", "musicals", "latin pop", "dream pop",
    "folk pop", "k-pop", "pop", "synthpop", "art pop", "soft pop", "acoustic pop",
    "power pop", "disco", "psychedelic pop", "jangle pop", "french pop",
    "variété française", "britpop", "post-disco", "colombian pop", "bachata pop",
    "norwegian pop", "europop", "hi-nrg", "bedroom pop", "desi pop", "brazilian pop",
    "italo disco", "flamenco pop", "punjabi pop", "hyperpop", "telugu pop",
    "marathi pop", "german pop", "indie pop", "c-pop", "hindi pop", "taiwanese pop",
    "bangla pop", "nederpop", "turkish pop", "j-pop", "kannada pop", "pop urbaine",
    "egyptian pop", "disco polo", "popular colombian music", "malaysian pop",
    "electropop", "t-pop", "bubblegum pop", "malayalam pop", "city pop", "dance pop",
    "ayalam pop", "gujarati pop", "indonesian pop", "funk pop", "finnish pop",
    "v-pop", "malay pop", "pop urbano", "haryanvi pop", "thai pop", "moroccan pop",
    "alternative pop", "bhojpuri pop"
],
    "religious": [
    "christian", "worship", "ccm", "gospel", "southern gospel", 
    "devotional", "khaleeji", "qawwali", "sufi", "ghazal", "mizrahi", 
    "gujarati garba", "african gospel", "brazilian evangelical music", "sholawat"
],
    "other": [
    "christmas", "soundtrack", "comedy", "children's music", "drone",
    "lullaby", "spoken word", "avant-garde", "experimental",
    "easy listening", "sandalwood", "tropical music"
]
}

def classify_genre(genre):
    genre = genre.lower().strip()
    
    for super_genre, subgenres in super_genres.items():
        if genre in subgenres:
            classification_stats[super_genre]["count"] += 1
            classification_stats[super_genre]["subgenres"][genre] += 1
            return super_genre

    
    classification_stats["other"]["count"] += 1
    classification_stats["other"]["subgenres"][genre] += 1
    return "other"

def print_classification_summary():
    print("\n🎵 Genre Classification Summary 🎵")
    for super_genre, stats in classification_stats.items():
        print(f"\n🔹 {super_genre.upper()} ({stats['count']} total tracks)")

       
        sorted_subgenres = sorted(stats["subgenres"].items(), key=lambda x: x[1], reverse=True)

        for subgenre, count in sorted_subgenres:
            print(f"   - {subgenre}: {count}")

    print("\n✅ Classification Complete!\n")



def plot_genre_treemap(classification_stats, base_font_size=3, min_count=0):

    supergenre_sizes = {sg: sum(stats["subgenres"].values()) for sg, stats in classification_stats.items()}

    
    supergenre_colors = sns.color_palette("tab20", n_colors=len(supergenre_sizes))
    supergenre_color_map = {sg: (r, g, b, 0.8) for i, (sg, (r, g, b)) in enumerate(zip(supergenre_sizes, supergenre_colors))}

 
    subgenre_labels = []
    subgenre_sizes = []
    subgenre_colors = []
    subgenre_counts = []

    supergenre_count_map = {sg: 0 for sg in supergenre_sizes}
    supergenre_subgenre_count_map = {sg: 0 for sg in supergenre_sizes} 

    
    for supergenre, stats in classification_stats.items():
        for subgenre, count in stats["subgenres"].items():
            if count >= min_count:
                subgenre_labels.append(subgenre)
                subgenre_sizes.append(count)  
                subgenre_colors.append(supergenre_color_map[supergenre])
                subgenre_counts.append(count)
                supergenre_count_map[supergenre] += 1  
                supergenre_subgenre_count_map[supergenre] += count  


    if not subgenre_counts:
        print("No subgenres meet the minimum count threshold.")
        return

    
    min_size, max_size = min(subgenre_counts), max(subgenre_counts)
    font_sizes = np.interp(subgenre_counts, [min_size, max_size], [base_font_size, base_font_size * 1.8])

  
    fig, ax = plt.subplots(figsize=(12, 18))

   
    rects = squarify.normalize_sizes(subgenre_sizes, 100, 100)
    rects = squarify.squarify(rects, 0, 0, 100, 100)


    for rect, label, color, fontsize in zip(rects, subgenre_labels, subgenre_colors, font_sizes):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        ax.add_patch(plt.Rectangle((x, y), dx, dy, facecolor=color, edgecolor="black", linewidth=0.5))
        ax.text(x + dx / 2, y + dy / 2, label, va='center', ha='center', fontsize=fontsize, wrap=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Supergenre Map", fontsize=14, fontweight="bold")

    
    num_genres = len(subgenre_labels)


    ax.text(
        0.05, -0.05,  
        f"Total Unique Subgenres: {num_genres}",
        ha='left', va='top', transform=ax.transAxes,
        fontsize=8, fontweight='normal'
    )

   
    print("\nSupergenre Subgenre Counts:")
    for supergenre, count in supergenre_subgenre_count_map.items():
        print(f"{supergenre}: {count} total count")

    handles = [plt.Rectangle((0, 0), 1, 1, color=supergenre_color_map[sg]) for sg in supergenre_sizes]
    plt.legend(handles, supergenre_sizes.keys(), title="Supergenres", loc="upper left", fontsize=8)
    
    # Show plot
    plt.show()

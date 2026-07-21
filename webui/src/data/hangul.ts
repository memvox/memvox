/** Hangul jamo reference data for Alphabet practice mode.
 *
 * `speak` is what gets sent to the pronunciation service: a bare jamo like
 * "ㄱ" is unpronounceable for a TTS voice, so consonants use their Korean
 * letter name (기역) and vowels their syllable form (아).
 */

export interface Jamo {
  glyph: string;
  /** Korean letter name (consonants) or syllable form (vowels). */
  name: string;
  romanization: string;
  /** Text handed to the pronunciation service. */
  speak: string;
  description: string;
  example: { word: string; meaning: string };
}

export interface JamoSection {
  title: string;
  blurb: string;
  letters: Jamo[];
}

export const HANGUL_SECTIONS: JamoSection[] = [
  {
    title: "Basic consonants",
    blurb: "The 14 basic consonants. Tap a letter for details and pronunciation.",
    letters: [
      { glyph: "ㄱ", name: "기역 (giyeok)", romanization: "g / k", speak: "기역", description: "Between English g and k — g at the start of a word, closer to k at the end.", example: { word: "가방", meaning: "bag" } },
      { glyph: "ㄴ", name: "니은 (nieun)", romanization: "n", speak: "니은", description: "Like English n.", example: { word: "나무", meaning: "tree" } },
      { glyph: "ㄷ", name: "디귿 (digeut)", romanization: "d / t", speak: "디귿", description: "Between English d and t — d between vowels, t at the end of a syllable.", example: { word: "다리", meaning: "leg, bridge" } },
      { glyph: "ㄹ", name: "리을 (rieul)", romanization: "r / l", speak: "리을", description: "A light tap like the tt in \"butter\"; sounds like l at the end of a syllable.", example: { word: "라면", meaning: "ramen" } },
      { glyph: "ㅁ", name: "미음 (mieum)", romanization: "m", speak: "미음", description: "Like English m.", example: { word: "물", meaning: "water" } },
      { glyph: "ㅂ", name: "비읍 (bieup)", romanization: "b / p", speak: "비읍", description: "Between English b and p — softer than either.", example: { word: "바다", meaning: "sea" } },
      { glyph: "ㅅ", name: "시옷 (siot)", romanization: "s", speak: "시옷", description: "Like s; becomes sh before the vowel ㅣ (시 = \"shi\").", example: { word: "사람", meaning: "person" } },
      { glyph: "ㅇ", name: "이응 (ieung)", romanization: "– / ng", speak: "이응", description: "Silent at the start of a syllable; ng (as in \"song\") at the end.", example: { word: "아이", meaning: "child" } },
      { glyph: "ㅈ", name: "지읒 (jieut)", romanization: "j", speak: "지읒", description: "Like English j, but with less lip rounding.", example: { word: "자다", meaning: "to sleep" } },
      { glyph: "ㅊ", name: "치읓 (chieut)", romanization: "ch", speak: "치읓", description: "Like English ch — an aspirated ㅈ.", example: { word: "차", meaning: "tea, car" } },
      { glyph: "ㅋ", name: "키읔 (kieuk)", romanization: "k", speak: "키읔", description: "A strongly aspirated k — ㄱ plus a puff of air.", example: { word: "코", meaning: "nose" } },
      { glyph: "ㅌ", name: "티읕 (tieut)", romanization: "t", speak: "티읕", description: "A strongly aspirated t — ㄷ plus a puff of air.", example: { word: "토끼", meaning: "rabbit" } },
      { glyph: "ㅍ", name: "피읖 (pieup)", romanization: "p", speak: "피읖", description: "A strongly aspirated p — ㅂ plus a puff of air.", example: { word: "포도", meaning: "grape" } },
      { glyph: "ㅎ", name: "히읗 (hieut)", romanization: "h", speak: "히읗", description: "Like English h.", example: { word: "하늘", meaning: "sky" } },
    ],
  },
  {
    title: "Tense (double) consonants",
    blurb: "Doubled letters pronounced with a tight, unaspirated burst.",
    letters: [
      { glyph: "ㄲ", name: "쌍기역 (ssanggiyeok)", romanization: "kk", speak: "쌍기역", description: "A tense, unaspirated k — throat tightened, no puff of air.", example: { word: "꿈", meaning: "dream" } },
      { glyph: "ㄸ", name: "쌍디귿 (ssangdigeut)", romanization: "tt", speak: "쌍디귿", description: "A tense, unaspirated t.", example: { word: "딸기", meaning: "strawberry" } },
      { glyph: "ㅃ", name: "쌍비읍 (ssangbieup)", romanization: "pp", speak: "쌍비읍", description: "A tense, unaspirated p.", example: { word: "빵", meaning: "bread" } },
      { glyph: "ㅆ", name: "쌍시옷 (ssangsiot)", romanization: "ss", speak: "쌍시옷", description: "A tense s with a sharper hiss than ㅅ.", example: { word: "쓰다", meaning: "to write" } },
      { glyph: "ㅉ", name: "쌍지읒 (ssangjieut)", romanization: "jj", speak: "쌍지읒", description: "A tense, unaspirated j.", example: { word: "찌개", meaning: "stew" } },
    ],
  },
  {
    title: "Basic vowels",
    blurb: "The 10 basic vowels, shown in their syllable form with silent ㅇ.",
    letters: [
      { glyph: "ㅏ", name: "아", romanization: "a", speak: "아", description: "Like the a in \"father\".", example: { word: "아빠", meaning: "dad" } },
      { glyph: "ㅑ", name: "야", romanization: "ya", speak: "야", description: "Like ya in \"yard\".", example: { word: "야구", meaning: "baseball" } },
      { glyph: "ㅓ", name: "어", romanization: "eo", speak: "어", description: "Like the u in \"gut\" — an open, unrounded o.", example: { word: "어머니", meaning: "mother" } },
      { glyph: "ㅕ", name: "여", romanization: "yeo", speak: "여", description: "y + ㅓ, like yu in \"yummy\" but more open.", example: { word: "여름", meaning: "summer" } },
      { glyph: "ㅗ", name: "오", romanization: "o", speak: "오", description: "Like the o in \"go\", with rounded lips.", example: { word: "오이", meaning: "cucumber" } },
      { glyph: "ㅛ", name: "요", romanization: "yo", speak: "요", description: "y + ㅗ, like yo in \"yoga\".", example: { word: "요리", meaning: "cooking" } },
      { glyph: "ㅜ", name: "우", romanization: "u", speak: "우", description: "Like the oo in \"moon\".", example: { word: "우유", meaning: "milk" } },
      { glyph: "ㅠ", name: "유", romanization: "yu", speak: "유", description: "y + ㅜ, like \"you\".", example: { word: "유리", meaning: "glass" } },
      { glyph: "ㅡ", name: "으", romanization: "eu", speak: "으", description: "No English equivalent — say \"oo\" with unrounded, spread lips.", example: { word: "그림", meaning: "picture" } },
      { glyph: "ㅣ", name: "이", romanization: "i", speak: "이", description: "Like the ee in \"see\".", example: { word: "이름", meaning: "name" } },
    ],
  },
  {
    title: "Compound vowels",
    blurb: "Combinations of basic vowels.",
    letters: [
      { glyph: "ㅐ", name: "애", romanization: "ae", speak: "애", description: "Like the e in \"bed\". In modern speech identical to ㅔ.", example: { word: "개", meaning: "dog" } },
      { glyph: "ㅒ", name: "얘", romanization: "yae", speak: "얘", description: "y + ㅐ.", example: { word: "얘기", meaning: "story, talk" } },
      { glyph: "ㅔ", name: "에", romanization: "e", speak: "에", description: "Like the e in \"bed\". In modern speech identical to ㅐ.", example: { word: "네", meaning: "yes" } },
      { glyph: "ㅖ", name: "예", romanization: "ye", speak: "예", description: "y + ㅔ, like ye in \"yes\".", example: { word: "예의", meaning: "manners" } },
      { glyph: "ㅘ", name: "와", romanization: "wa", speak: "와", description: "Like wa in \"water\".", example: { word: "과일", meaning: "fruit" } },
      { glyph: "ㅙ", name: "왜", romanization: "wae", speak: "왜", description: "Like we in \"wet\".", example: { word: "왜", meaning: "why" } },
      { glyph: "ㅚ", name: "외", romanization: "oe", speak: "외", description: "Pronounced like ㅙ in modern speech (\"we\").", example: { word: "회사", meaning: "company" } },
      { glyph: "ㅝ", name: "워", romanization: "wo", speak: "워", description: "Like wo in \"wonder\".", example: { word: "원", meaning: "won (₩)" } },
      { glyph: "ㅞ", name: "웨", romanization: "we", speak: "웨", description: "Like we in \"wet\"; rare.", example: { word: "웨딩", meaning: "wedding" } },
      { glyph: "ㅟ", name: "위", romanization: "wi", speak: "위", description: "Like wee in \"week\".", example: { word: "귀", meaning: "ear" } },
      { glyph: "ㅢ", name: "의", romanization: "ui", speak: "의", description: "ㅡ gliding into ㅣ; often reduces to \"e\" or \"i\" in speech.", example: { word: "의사", meaning: "doctor" } },
    ],
  },
];

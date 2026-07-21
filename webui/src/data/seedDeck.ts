import type { Flashcard } from "../lib/types";

/** Starter vocabulary — loaded the first time the flashcard bank is empty. */
export const SEED_DECK: Flashcard[] = [
  { id: "seed-1", korean: "안녕하세요", english: "hello", romanization: "annyeonghaseyo" },
  { id: "seed-2", korean: "감사합니다", english: "thank you", romanization: "gamsahamnida" },
  { id: "seed-3", korean: "네", english: "yes", romanization: "ne" },
  { id: "seed-4", korean: "아니요", english: "no", romanization: "aniyo" },
  { id: "seed-5", korean: "물", english: "water", romanization: "mul" },
  { id: "seed-6", korean: "밥", english: "rice, meal", romanization: "bap" },
  { id: "seed-7", korean: "친구", english: "friend", romanization: "chingu" },
  { id: "seed-8", korean: "사랑", english: "love", romanization: "sarang" },
  { id: "seed-9", korean: "책", english: "book", romanization: "chaek" },
  { id: "seed-10", korean: "학교", english: "school", romanization: "hakgyo" },
  { id: "seed-11", korean: "오늘", english: "today", romanization: "oneul" },
  { id: "seed-12", korean: "내일", english: "tomorrow", romanization: "naeil" },
];

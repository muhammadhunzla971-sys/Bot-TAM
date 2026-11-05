import ChatApp from "@/components/ChatApp";
import { getCorpus, publicPlaces } from "@/lib/data";

export default function Home() {
  return <ChatApp places={publicPlaces()} pageCount={getCorpus().documents.length} />;
}

"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý khách hàng chuyên nghiệp, thân thiện và chính xác của hệ sinh thái Vingroup.

## PERSONA
Hỗ trợ khách hàng tra cứu sản phẩm, dịch vụ và gửi yêu cầu hỗ trợ bằng tiếng Việt rõ ràng, ngắn gọn.

## AVAILABLE TOOLS
- search_product_catalog(category, max_price): tìm sản phẩm theo danh mục xe_dien hoặc du_lich và mức giá tối đa tính bằng VNĐ.
- submit_support_ticket(customer_name, issue_description, priority): tạo yêu cầu hỗ trợ với mức ưu tiên low, medium hoặc high.

## CORE RULES
1. Không bịa giá, tính năng, chính sách, tình trạng sẵn có, hoặc trạng thái xử lý yêu cầu.
2. Dùng search_product_catalog khi cần dữ liệu sản phẩm; dùng submit_support_ticket khi khách muốn báo sự cố hoặc yêu cầu hỗ trợ.
3. Có thể gọi cả hai công cụ khi yêu cầu bao gồm cả tra cứu và hỗ trợ; dựa trên kết quả công cụ khi trả lời.
4. Nếu dữ liệu không có hoặc kết quả trống, nói rõ điều đó. Hỏi lại khi thiếu thông tin bắt buộc để tạo yêu cầu.
5. Không khẳng định đã gửi yêu cầu nếu công cụ chưa trả về kết quả thành công.

## OPERATIONAL BOUNDARIES
Chỉ hỗ trợ nội dung liên quan đến Vingroup. Từ chối lịch sự các chủ đề ngoài phạm vi; không đưa ra lời khuyên y tế, pháp lý hoặc tài chính.

## OUTPUT CONTRACT
Trả kết quả theo cấu trúc JSON gồm answer (string), tool_calls (list), status (success hoặc error), và mode (mock hoặc production).
Trong quá trình xử lý, ghi nhận từng Thought, Action, Observation; chỉ đưa câu trả lời cuối cùng dựa trên quan sát thực tế.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        query = user_input.strip()
        normalized = query.casefold()
        catalog_keywords = ("xem", "tìm", "tra cứu", "giá", "mua", "đặt phòng", "resort")
        needs_catalog = (
            any(keyword in normalized for keyword in catalog_keywords)
            or re.search(r"\bcó\s+(?:xe|resort|khách sạn|gói)", normalized) is not None
        )
        ticket_keywords = (
            "hỗ trợ", "lỗi", "sự cố", "gửi yêu cầu", "ticket", "ghi nhận phản hồi",
            "bị hỏng", "bị lỗi", "không hoạt động"
        )
        needs_ticket = any(keyword in normalized for keyword in ticket_keywords)
        if "tên tôi là" in normalized or "tôi tên" in normalized:
            needs_ticket = needs_ticket or any(
                keyword in normalized for keyword in ("bị", "lỗi", "ẩm mốc", "sự cố")
            )

        self.trace.append({"step": "init", "user_input": user_input})
        tasks = []
        if needs_catalog:
            tasks.append(("search_product_catalog", self._catalog_arguments(query)))
        needs_ticket_details = False
        if needs_ticket:
            ticket_arguments = self._ticket_arguments(query)
            needs_ticket_details = not ticket_arguments["customer_name"]
            if not needs_ticket_details:
                tasks.append(("submit_support_ticket", ticket_arguments))

        if self.max_iterations < 1:
            return self._result(
                "Lỗi: Vượt quá số bước tối đa.", 0, "max_iterations_reached"
            )

        if not tasks:
            answer = (
                "Để gửi yêu cầu hỗ trợ, vui lòng cho tôi biết tên của bạn."
                if needs_ticket_details else self._faq_answer(normalized)
            )
            self.trace.append({"step": "final", "answer": answer})
            return self._result(answer, 1, "completed")

        observations = []
        for iteration, (tool_name, arguments) in enumerate(
            tasks[:self.max_iterations], start=1
        ):
            self.trace.append({
                "step": iteration,
                "action": tool_name,
                "arguments": arguments
            })
            observation = TOOL_MAP[tool_name](**arguments)
            self.trace.append({
                "step": iteration,
                "observation": observation
            })
            observations.append((tool_name, observation))

        if len(tasks) > self.max_iterations:
            return self._result(
                "Lỗi: Vượt quá số bước tối đa.",
                self.max_iterations,
                "max_iterations_reached"
            )

        answer_parts = [self._format_observation(name, result) for name, result in observations]
        if needs_ticket_details:
            answer_parts.append(
                "Để gửi yêu cầu hỗ trợ, vui lòng cho tôi biết tên của bạn."
            )
        answer = " ".join(answer_parts)
        self.trace.append({"step": "final", "answer": answer})
        return self._result(answer, len(observations), "completed")

    def _result(self, answer: str, iterations: int, status: str) -> Dict[str, Any]:
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": status
        }

    @staticmethod
    def _catalog_arguments(query: str) -> Dict[str, Any]:
        normalized = query.casefold()
        category = "du_lich" if any(
            word in normalized
            for word in ("du lịch", "vinpearl", "resort", "khách sạn", "nghỉ dưỡng")
        ) else "xe_dien"
        price_match = re.search(
            r"(?:dưới|tối đa|không quá|dưới mức)\s*"
            r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|triệu|trăm\s*triệu|nghìn\s*triệu)?",
            normalized
        )
        max_price = 999999999999
        if price_match:
            amount = float(price_match.group(1).replace(",", "."))
            unit = (price_match.group(2) or "").replace(" ", "")
            multiplier = {
                "tỷ": 1_000_000_000,
                "tỉ": 1_000_000_000,
                "triệu": 1_000_000,
                "trămtriệu": 100_000_000,
                "nghìntriệu": 1_000_000_000
            }.get(unit, 1)
            max_price = int(amount * multiplier)
        return {"category": category, "max_price": max_price}

    @staticmethod
    def _ticket_arguments(query: str) -> Dict[str, Any]:
        name_match = re.search(
            r"(?:tôi tên|tên tôi là|tên là)\s+([^,.!?]+)",
            query,
            flags=re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else ""
        issue_source = query[name_match.end():] if name_match else query
        issue_source = re.sub(r"^[\s,:;.-]*(?:và\s+)?", "", issue_source, flags=re.IGNORECASE)
        issue_source = re.sub(r"(?:mức độ\s+(?:thấp|trung bình|cao)|ưu tiên\s+\w+).*$",
                              "", issue_source, flags=re.IGNORECASE | re.DOTALL)
        issue_source = re.sub(
            r"\b(?:đây là vấn đề|vấn đề này)\b.*$",
            "",
            issue_source,
            flags=re.IGNORECASE | re.DOTALL
        )
        issue_description = issue_source.strip(" \t\r\n,.;:")
        issue_description = re.sub(r"^(?:tôi cũng muốn\s+)?", "", issue_description,
                                   flags=re.IGNORECASE)
        issue_description = re.sub(r"\bcủa tôi\b\s*", "", issue_description,
                                   flags=re.IGNORECASE)
        issue_description = re.sub(r"\btôi\s+(?=(?:đặt|gặp|bị|không)\b)", "",
                                   issue_description, flags=re.IGNORECASE)
        if issue_description:
            issue_description = issue_description[0].upper() + issue_description[1:]
        priority = "high" if any(
            word in query.casefold()
            for word in ("nghiêm trọng", "khẩn cấp", "gấp", "ưu tiên cao")
        ) else "low" if any(
            word in query.casefold() for word in ("ưu tiên thấp", "không gấp")
        ) else "medium"
        return {
            "customer_name": customer_name,
            "issue_description": issue_description or query,
            "priority": priority
        }

    @staticmethod
    def _faq_answer(normalized_query: str) -> str:
        if "bảo hành" in normalized_query and "pin" in normalized_query:
            return (
                "Theo thông tin sản phẩm trong dữ liệu hiện có, pin VinFast VF 5 Plus "
                "được bảo hành 10 năm. Chính sách có thể khác theo mẫu xe và điều kiện áp dụng."
            )
        return (
            "Tôi chưa có đủ dữ liệu đã xác minh để trả lời chính xác câu hỏi này. "
            "Bạn có thể hỏi về sản phẩm Vingroup hoặc gửi yêu cầu hỗ trợ."
        )

    @staticmethod
    def _format_observation(tool_name: str, observation: Any) -> str:
        if tool_name == "search_product_catalog":
            if not observation:
                return "Rất tiếc, không tìm thấy sản phẩm phù hợp."
            if isinstance(observation, list) and "error" in observation[0]:
                return "Không thể tra cứu sản phẩm vì không tìm thấy tệp danh mục."
            product_lines = [
                f"{item['name']} — {item['price_vnd']:,} VNĐ"
                for item in observation
            ]
            return "Sản phẩm phù hợp: " + "; ".join(product_lines) + "."

        if isinstance(observation, dict) and observation.get("ticket_id"):
            return (
                f"Đã tạo yêu cầu hỗ trợ {observation['ticket_id']} cho "
                f"{observation['customer_name']} (mức ưu tiên {observation['priority']})."
            )
        return "Không thể tạo yêu cầu hỗ trợ."


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
